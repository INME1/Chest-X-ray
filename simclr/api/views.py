from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import requests, os, json
import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import io
import pydicom

from ..ml.models import SimCLR, MultiScaleModelEnsemble

# 🔧 Orthanc 설정
ORTHANC_URL = 'http://35.225.63.41:8042'
AUTH = ('orthanc', 'orthanc')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patch_sizes = [128, 256, 512]
model_paths = [
    os.path.join(settings.BASE_DIR, 'simclr', 'weights', f'simclr_{patch_size}px.pth') 
    for patch_size in patch_sizes
]
reference_dirs = [
    os.path.join(settings.BASE_DIR, 'simclr', 'references', f'patch{patch_size}')
    for patch_size in patch_sizes
]

try:
    ensemble_model = MultiScaleModelEnsemble(model_paths, patch_sizes, device=device)
    model_loaded = True
    logger.info("✅ 이상 탐지 모델 로드 성공")
except Exception as e:
    logger.exception("❌ 이상 탐지 모델 로드 실패")
    model_loaded = False

@csrf_exempt
def check_processed_view(request, instance_uid):
    result_dir = os.path.join(settings.MEDIA_ROOT, 'anomaly_results', instance_uid)
    if os.path.exists(result_dir) and os.path.isfile(os.path.join(result_dir, 'overlay.png')):
        media_url = settings.MEDIA_URL
        meta_path = os.path.join(result_dir, 'metadata.json')
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                try:
                    metadata = json.load(f)
                except:
                    pass
        return JsonResponse({
            "processed": True,
            "original_url": f"{media_url}anomaly_results/{instance_uid}/original.png",
            "heatmap_url": f"{media_url}anomaly_results/{instance_uid}/heatmap.png",
            "overlay_url": f"{media_url}anomaly_results/{instance_uid}/overlay.png",
            "score_avg": metadata.get("score_avg", 0),
            "score_max": metadata.get("score_max", 0),
            "score_percentile_95": metadata.get("score_percentile_95", 0),
            "is_anomalous": metadata.get("is_anomalous", False)
        })
    else:
        return JsonResponse({"processed": False})

def find_internal_id_by_sop_uid(sop_uid):
    response = requests.get(f"{ORTHANC_URL}/instances", auth=AUTH)
    for iid in response.json():
        meta = requests.get(f"{ORTHANC_URL}/instances/{iid}", auth=AUTH).json()
        dicom_sop_uid = meta.get("MainDicomTags", {}).get("SOPInstanceUID")
        if dicom_sop_uid == sop_uid:
            return iid
    return None

@csrf_exempt
def process_anomaly_view(request, instance_uid):
    if not model_loaded:
        return JsonResponse({"status": "error", "message": "이상 탐지 모델 로드 실패"}, status=500)

    try:
        # ✅ 내부 ID 찾기: SOPInstanceUID → Orthanc 내부 instance ID
        instance_list = requests.get(f"{ORTHANC_URL}/instances", auth=AUTH).json()
        matched_instance_id = None

        for iid in instance_list:
            meta = requests.get(f"{ORTHANC_URL}/instances/{iid}", auth=AUTH).json()
            dicom_sop_uid = meta.get("MainDicomTags", {}).get("SOPInstanceUID")
            if dicom_sop_uid == instance_uid:
                matched_instance_id = iid
                break

        if not matched_instance_id:
            return JsonResponse({
                "status": "error",
                "message": f"SOPInstanceUID {instance_uid}에 해당하는 인스턴스를 찾을 수 없습니다."
            }, status=404)

        # ✅ DICOM 다운로드
        dicom_url = f"{ORTHANC_URL}/instances/{matched_instance_id}/file"
        dicom_response = requests.get(dicom_url, auth=AUTH)
        if dicom_response.status_code != 200:
            return JsonResponse({
                "status": "error",
                "message": f"DICOM 데이터 가져오기 실패: {dicom_response.status_code}"
            }, status=500)

        dicom_data = pydicom.dcmread(io.BytesIO(dicom_response.content))
        img = dicom_to_image(dicom_data)

        # ✅ 임시 저장 및 분석
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{instance_uid}.png")
        img.save(temp_path)

        anomaly_map, _ = ensemble_model.generate_anomaly_map(temp_path, reference_dirs)

        result_dir = os.path.join(settings.MEDIA_ROOT, 'anomaly_results', instance_uid)
        os.makedirs(result_dir, exist_ok=True)
        img.save(os.path.join(result_dir, 'original.png'))

        anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (np.ptp(anomaly_map) + 1e-8)
        heatmap = (cm.jet(anomaly_map_normalized)[:, :, :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap)
        heatmap_img.save(os.path.join(result_dir, 'heatmap.png'))

        overlay = Image.blend(img, heatmap_img, alpha=0.5)
        overlay.save(os.path.join(result_dir, 'overlay.png'))

        metrics = calculate_anomaly_metrics(anomaly_map)
        with open(os.path.join(result_dir, 'metadata.json'), 'w') as f:
            json.dump(metrics, f)

        try:
            os.remove(temp_path)
        except:
            pass

        media_url = settings.MEDIA_URL
        return JsonResponse({
            "status": "success",
            "instance_id": instance_uid,
            "original_url": f"{media_url}anomaly_results/{instance_uid}/original.png",
            "heatmap_url": f"{media_url}anomaly_results/{instance_uid}/heatmap.png",
            "overlay_url": f"{media_url}anomaly_results/{instance_uid}/overlay.png",
            **metrics
        })

    except Exception as e:
        logger.exception("이상 탐지 처리 실패")
        return JsonResponse({
            "status": "error",
            "message": f"이상 탐지 처리 실패: {str(e)}"
        }, status=500)

def dicom_to_image(dicom_data):
    pixel_array = dicom_data.pixel_array
    if hasattr(dicom_data, "WindowCenter") and hasattr(dicom_data, "WindowWidth"):
        center = dicom_data.WindowCenter
        width = dicom_data.WindowWidth
        if isinstance(center, list):
            center = center[0]
        if isinstance(width, list):
            width = width[0]
        min_value = center - width // 2
        max_value = center + width // 2
        pixel_array = np.clip(pixel_array, min_value, max_value)
    if pixel_array.max() > 0:
        pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())) * 255.0
    pixel_array = pixel_array.astype(np.uint8)
    if len(pixel_array.shape) == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=2)
    return Image.fromarray(pixel_array)

def calculate_anomaly_metrics(anomaly_map):
    score_avg = float(np.mean(anomaly_map))
    score_max = float(np.max(anomaly_map))
    score_percentile_95 = float(np.percentile(anomaly_map, 95))
    threshold = score_percentile_95 * 0.7
    is_anomalous = bool(score_max > threshold)
    return {
        'score_avg': score_avg,
        'score_max': score_max,
        'score_percentile_95': score_percentile_95,
        'is_anomalous': is_anomalous
    }

@csrf_exempt
def get_first_instance_from_study(request, study_uid):
    try:
        logger.info(f"🔍 요청된 StudyInstanceUID: {study_uid}")
        series_url = f"{ORTHANC_URL}/dicom-web/studies/{study_uid}/series"
        series_resp = requests.get(series_url, auth=AUTH, headers={'Accept': 'application/dicom+json'})
        if series_resp.status_code != 200:
            return JsonResponse({"status": "error", "message": f"시리즈 목록 조회 실패: {series_resp.status_code}"}, status=series_resp.status_code)
        series_list = series_resp.json()
        if not series_list:
            return JsonResponse({"status": "error", "message": "해당 Study에 Series가 없습니다."}, status=404)

        first_series = series_list[0]
        first_series_uid = first_series.get("0020000E", {}).get("Value", [None])[0]
        if not first_series_uid:
            return JsonResponse({"status": "error", "message": "SeriesInstanceUID를 찾을 수 없습니다."}, status=500)

        instances_url = f"{ORTHANC_URL}/dicom-web/series/{first_series_uid}/instances"
        instances_resp = requests.get(instances_url, auth=AUTH, headers={'Accept': 'application/dicom+json'})
        if instances_resp.status_code != 200:
            return JsonResponse({"status": "error", "message": f"인스턴스 목록 조회 실패: {instances_resp.status_code}"}, status=instances_resp.status_code)
        instances = instances_resp.json()
        if not instances:
            return JsonResponse({"status": "error", "message": "해당 Series에 인스턴스가 없습니다."}, status=404)

        first_instance = instances[0]
        sop_instance_uid = first_instance.get("00080018", {}).get("Value", [None])[0]

        return JsonResponse({
            "status": "success",
            "study_uid": study_uid,
            "first_series_uid": first_series_uid,
            "first_instance_uid": sop_instance_uid,
            "instance_info": first_instance
        })

    except Exception as e:
        logger.exception("❌ 첫 번째 인스턴스 조회 중 오류 발생")
        return JsonResponse({"status": "error", "message": f"서버 오류: {str(e)}"}, status=500)

@csrf_exempt
def list_study_metadata(request):
    try:
        studies = requests.get(f"{ORTHANC_URL}/studies", auth=AUTH).json()
        result = []
        for sid in studies:
            r = requests.get(f"{ORTHANC_URL}/studies/{sid}", auth=AUTH)
            if r.status_code != 200:
                continue
            meta = r.json()
            result.append({
                "0020000D": {"Value": [meta.get("MainDicomTags", {}).get("StudyInstanceUID", "")]},
                "00100010": {"Value": [{"Alphabetic": meta.get("MainDicomTags", {}).get("PatientName", "Unknown")}]},
                "00080020": {"Value": [meta.get("MainDicomTags", {}).get("StudyDate", "")]}
            })
        return JsonResponse(result, safe=False)
    except Exception as e:
        logger.exception("❌ 스터디 메타데이터 조회 실패")
        return JsonResponse({"error": f"스터디 목록 조회 실패: {str(e)}"}, status=500)
