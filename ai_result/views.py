from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import requests, os, json
import logging

# 🔧 Orthanc 설정
ORTHANC_URL = 'http://35.225.63.41:8042'
AUTH = ('orthanc', 'orthanc')
logger = logging.getLogger(__name__)

# ✅ ① DICOM 업로드
@csrf_exempt
@require_http_methods(["POST"])
def upload_dicom(request):
    if request.FILES.get('file'):
        dicom_file = request.FILES['file']
        logger.info(f"📥 업로드 파일명: {dicom_file.name}")
        try:
            response = requests.post(
                f"{ORTHANC_URL}/instances",
                auth=AUTH,
                files={'file': (dicom_file.name, dicom_file.read())}
            )
            logger.info(f"📤 업로드 상태: {response.status_code}")
            try:
                return JsonResponse(response.json(), status=response.status_code)
            except ValueError:
                return JsonResponse({"message": "DICOM 업로드 성공 (응답 없음)"}, status=200)
        except Exception as e:
            logger.exception("❌ DICOM 업로드 실패")
            return JsonResponse({'error': 'Upload failed'}, status=500)
    return JsonResponse({'error': 'DICOM 파일이 필요합니다.'}, status=400)


# ✅ ② 환자 목록
def patient_list_view(request):
    try:
        response = requests.get(f"{ORTHANC_URL}/patients/", auth=AUTH)
        return JsonResponse(response.json(), safe=False)
    except Exception as e:
        logger.exception("환자 목록 조회 실패")
        return JsonResponse({'error': str(e)}, status=500)


# ✅ ③ 스터디 목록
def study_list_view(request):
    try:
        response = requests.get(f"{ORTHANC_URL}/studies/", auth=AUTH)
        return JsonResponse(response.json(), safe=False)
    except Exception as e:
        logger.exception("스터디 목록 조회 실패")
        return JsonResponse({'error': str(e)}, status=500)


# ✅ ④ 스터디 상세정보
def study_detail_view(request, study_uid):
    try:
        response = requests.get(f"{ORTHANC_URL}/studies/{study_uid}", auth=AUTH)
        if response.status_code == 200:
            return JsonResponse(response.json(), status=200)
        return JsonResponse({'error': 'Study not found'}, status=404)
    except Exception as e:
        logger.exception("스터디 상세조회 실패")
        return JsonResponse({'error': str(e)}, status=500)


# ✅ ⑤ Series 리스트 (스터디 기준)
def series_list_by_study(request, study_uid):
    try:
        response = requests.get(f'{ORTHANC_URL}/studies/{study_uid}/series', auth=AUTH)
        if response.status_code == 200:
            return JsonResponse(response.json(), safe=False)
        return JsonResponse({'error': 'Series not found'}, status=404)
    except Exception as e:
        logger.exception("시리즈 조회 실패")
        return JsonResponse({'error': str(e)}, status=500)


# ✅ ⑥ Instance 리스트 (시리즈 기준)
def instance_list_by_series(request, series_uid):
    try:
        response = requests.get(f'{ORTHANC_URL}/series/{series_uid}/instances', auth=AUTH)
        if response.status_code == 200:
            return JsonResponse(response.json(), safe=False)
        return JsonResponse({'error': 'Instances not found'}, status=404)
    except Exception as e:
        logger.exception("인스턴스 조회 실패")
        return JsonResponse({'error': str(e)}, status=500)


# ✅ ⑦ OHIF용 프록시 (DICOMWeb redirect)
@csrf_exempt
def dicom_proxy(request):
    relative_path = request.get_full_path().replace('/dicom-web', '', 1)
    if relative_path in ['', '/']:
        return HttpResponse("Bad request", status=400)

    url = f'{ORTHANC_URL}{relative_path}'
    headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}

    try:
        response = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            params=request.GET,
            data=request.body,
            auth=AUTH,
            stream=True,
        )
        excluded = ['connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
                    'te', 'trailers', 'transfer-encoding', 'upgrade']
        filtered = {k: v for k, v in response.headers.items() if k.lower() not in excluded}
        proxy_response = HttpResponse(response.content, status=response.status_code)
        for k, v in filtered.items():
            proxy_response[k] = v
        return proxy_response
    except Exception as e:
        logger.exception("📡 DICOM 프록시 오류")
        return HttpResponse(f"Proxy error: {str(e)}", status=500)


# ✅ ⑧ AI 결과 반환
@csrf_exempt
def model_results_view(request, study_uid):
    results_dir = os.path.join(settings.MEDIA_ROOT, 'ai_results', study_uid)
    results = []

    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    meta = json.load(f)
                    results.append(meta)
            elif filename.endswith(('.png', '.jpg')):
                model_name = filename.replace('_result.png', '').replace('.png', '')
                results.append({
                    "model": model_name,
                    "image_url": f"/media/ai_results/{study_uid}/{filename}",
                    "probability": None
                })

    return JsonResponse(results, safe=False)
from django.http import JsonResponse
import requests

@csrf_exempt
def single_study_view(request, study_uid):
    try:
        # 1. Series 목록 가져오기
        series_url = f"{ORTHANC_URL}/dicom-web/studies/{study_uid}/series"
        series_resp = requests.get(series_url, auth=AUTH, headers={'Accept': 'application/dicom+json'})
        series_list = series_resp.json()

        full_response = []

        for series in series_list:
            series_uid = series.get("0020000E", {}).get("Value", [None])[0]
            if not series_uid:
                continue

            # 2. 이 series 안의 인스턴스들 가져오기
            instances_url = f"{ORTHANC_URL}/dicom-web/series/{series_uid}/instances"
            inst_resp = requests.get(instances_url, auth=AUTH, headers={'Accept': 'application/dicom+json'})
            instances = inst_resp.json()

            full_response.extend(instances)

        return JsonResponse(full_response, safe=False)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def test_endpoint(request):
    """가장 간단한 테스트 엔드포인트"""
    return JsonResponse({"status": "ok", "message": "테스트 엔드포인트가 작동합니다"})
