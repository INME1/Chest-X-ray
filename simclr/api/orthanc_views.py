# simclr/api/orthanc_views.py
"""
Orthanc PACS 서버와 연동을 위한 API 엔드포인트
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import requests
import logging
import json
import os
import io
import pydicom
import numpy as np
from PIL import Image
from django.conf import settings
import base64

logger = logging.getLogger(__name__)

# Orthanc 설정
ORTHANC_URL = 'http://localhost:8042'
AUTH = ("orthanc", "orthanc")

@csrf_exempt
@require_http_methods(["GET"])
def get_study_list(request):
    """
    Orthanc 서버에서 최근 연구(Studies) 목록 가져오기
    """
    try:
        # 최근 연구 가져오기 (최대 20개)
        response = requests.get(f"{ORTHANC_URL}/studies", auth=AUTH)
        if response.status_code != 200:
            return JsonResponse({
                "status": "error",
                "message": f"Orthanc 서버 연결 실패: {response.status_code}"
            }, status=500)
        
        study_ids = response.json()[:20]  # 최대 20개
        studies = []
        
        # 각 연구의 세부 정보 가져오기
        for study_id in study_ids:
            study_info = requests.get(f"{ORTHANC_URL}/studies/{study_id}", auth=AUTH).json()
            
            if 'MainDicomTags' in study_info:
                tags = study_info['MainDicomTags']
                
                # 첫 번째 인스턴스 ID 가져오기 (썸네일용)
                first_instance_id = None
                if 'Instances' in study_info and study_info['Instances']:
                    first_instance_id = study_info['Instances'][0]
                
                study_data = {
                    "studyID": study_id,
                    "patientName": tags.get('PatientName', 'Unknown'),
                    "studyDate": tags.get('StudyDate', 'Unknown'),
                    "studyDescription": tags.get('StudyDescription', 'No description'),
                    "seriesCount": len(study_info.get('Series', [])),
                    "instancesCount": len(study_info.get('Instances', [])),
                    "previewInstanceID": first_instance_id
                }
                studies.append(study_data)
        
        return JsonResponse({
            "status": "success",
            "studies": studies
        })
    
    except Exception as e:
        logger.exception("연구 목록 가져오기 실패")
        return JsonResponse({
            "status": "error",
            "message": f"연구 목록 가져오기 실패: {str(e)}"
        }, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def get_series_list(request, study_id):
    """
    특정 연구(Study)의 시리즈(Series) 목록 가져오기
    """
    try:
        # 연구에 속한 시리즈 ID 목록 가져오기
        response = requests.get(f"{ORTHANC_URL}/studies/{study_id}/series", auth=AUTH)
        if response.status_code != 200:
            return JsonResponse({
                "status": "error",
                "message": f"시리즈 목록 가져오기 실패: {response.status_code}"
            }, status=500)
        
        series_ids = response.json()
        series_list = []
        
        # 각 시리즈의 세부 정보 가져오기
        for series_id in series_ids:
            series_info = requests.get(f"{ORTHANC_URL}/series/{series_id}", auth=AUTH).json()
            
            if 'MainDicomTags' in series_info:
                tags = series_info['MainDicomTags']
                
                # 첫 번째 인스턴스 ID 가져오기 (썸네일용)
                first_instance_id = None
                if 'Instances' in series_info and series_info['Instances']:
                    first_instance_id = series_info['Instances'][0]
                
                series_data = {
                    "seriesID": series_id,
                    "seriesDescription": tags.get('SeriesDescription', 'No description'),
                    "modality": tags.get('Modality', 'Unknown'),
                    "seriesNumber": tags.get('SeriesNumber', 'Unknown'),
                    "instancesCount": len(series_info.get('Instances', [])),
                    "previewInstanceID": first_instance_id
                }
                series_list.append(series_data)
        
        return JsonResponse({
            "status": "success",
            "studyID": study_id,
            "series": series_list
        })
    
    except Exception as e:
        logger.exception("시리즈 목록 가져오기 실패")
        return JsonResponse({
            "status": "error",
            "message": f"시리즈 목록 가져오기 실패: {str(e)}"
        }, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def get_instances_list(request, series_id):
    """
    특정 시리즈(Series)의 인스턴스(Instances) 목록 가져오기
    """
    try:
        # 시리즈에 속한 인스턴스 ID 목록 가져오기
        response = requests.get(f"{ORTHANC_URL}/series/{series_id}/instances", auth=AUTH)
        if response.status_code != 200:
            return JsonResponse({
                "status": "error",
                "message": f"인스턴스 목록 가져오기 실패: {response.status_code}"
            }, status=500)
        
        instance_ids = response.json()
        instances_list = []
        
        # 각 인스턴스의 세부 정보 가져오기
        for instance_id in instance_ids:
            instance_info = requests.get(f"{ORTHANC_URL}/instances/{instance_id}", auth=AUTH).json()
            
            if 'MainDicomTags' in instance_info:
                tags = instance_info['MainDicomTags']
                
                # 인스턴스 썸네일 URL 생성
                thumbnail_url = f"{ORTHANC_URL}/instances/{instance_id}/preview"
                
                instance_data = {
                    "instanceID": instance_id,
                    "sopInstanceUID": tags.get('SOPInstanceUID', 'Unknown'),
                    "instanceNumber": tags.get('InstanceNumber', 'Unknown'),
                    "thumbnailUrl": thumbnail_url
                }
                instances_list.append(instance_data)
        
        # 인스턴스 번호 기준으로 정렬
        instances_list.sort(key=lambda x: int(x['instanceNumber']) if x['instanceNumber'].isdigit() else 9999)
        
        return JsonResponse({
            "status": "success",
            "seriesID": series_id,
            "instances": instances_list
        })
    
    except Exception as e:
        logger.exception("인스턴스 목록 가져오기 실패")
        return JsonResponse({
            "status": "error",
            "message": f"인스턴스 목록 가져오기 실패: {str(e)}"
        }, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def get_instance_preview(request, instance_id):
    """
    인스턴스 미리보기 이미지 가져오기 (Base64 인코딩)
    """
    try:
        # Orthanc 서버에서 DICOM 파일 가져오기
        response = requests.get(f"{ORTHANC_URL}/instances/{instance_id}/file", auth=AUTH)
        if response.status_code != 200:
            return JsonResponse({
                "status": "error",
                "message": f"DICOM 파일 가져오기 실패: {response.status_code}"
            }, status=500)
        
        # DICOM 파싱
        dicom_data = pydicom.dcmread(io.BytesIO(response.content))
        
        # 픽셀 배열 추출 및 이미지 변환
        pixel_array = dicom_data.pixel_array
        
        # 정규화
        if pixel_array.max() > 0:
            pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())) * 255.0
        
        # 8비트 변환
        pixel_array = pixel_array.astype(np.uint8)
        
        # 단일 채널인 경우 RGB로 변환
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack([pixel_array] * 3, axis=2)
        
        # PIL 이미지로 변환
        img = Image.fromarray(pixel_array)
        
        # 크기 조정 (미리보기용)
        img.thumbnail((256, 256), Image.LANCZOS)
        
        # Base64 인코딩
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return JsonResponse({
            "status": "success",
            "instanceID": instance_id,
            "previewBase64": f"data:image/png;base64,{img_str}"
        })
    
    except Exception as e:
        logger.exception("인스턴스 미리보기 가져오기 실패")
        return JsonResponse({
            "status": "error",
            "message": f"인스턴스 미리보기 가져오기 실패: {str(e)}"
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def search_instances(request):
    """
    DICOM 인스턴스 검색
    """
    try:
        data = json.loads(request.body)
        query = data.get('query', '')
        modality = data.get('modality', 'CR')  # 기본값: CR (X-Ray)
        limit = data.get('limit', 20)
        
        # Orthanc의 모든 연구 가져오기
        response = requests.get(f"{ORTHANC_URL}/studies", auth=AUTH)
        if response.status_code != 200:
            return JsonResponse({
                "status": "error",
                "message": f"연구 목록 가져오기 실패: {response.status_code}"
            }, status=500)
        
        study_ids = response.json()
        results = []
        
        # 각 연구에서 X-Ray 이미지 검색
        for study_id in study_ids[:50]:  # 최대 50개 연구만 검색
            try:
                # 연구 정보 가져오기
                study_info = requests.get(f"{ORTHANC_URL}/studies/{study_id}", auth=AUTH).json()
                patient_name = study_info.get('MainDicomTags', {}).get('PatientName', '')
                study_date = study_info.get('MainDicomTags', {}).get('StudyDate', '')
                
                # 쿼리가 없거나 환자 이름에 쿼리가 포함되어 있으면 계속 진행
                if not query or query.lower() in patient_name.lower():
                    # 연구에 속한 시리즈 가져오기
                    series_response = requests.get(f"{ORTHANC_URL}/studies/{study_id}/series", auth=AUTH)
                    series_ids = series_response.json()
                    
                    for series_id in series_ids:
                        # 시리즈 정보 가져오기
                        series_info = requests.get(f"{ORTHANC_URL}/series/{series_id}", auth=AUTH).json()
                        
                        # 모달리티가 CR(X-Ray)인 시리즈만 선택
                        if series_info.get('MainDicomTags', {}).get('Modality') == modality:
                            # 시리즈의 첫 번째 인스턴스 가져오기
                            if 'Instances' in series_info and series_info['Instances']:
                                instance_id = series_info['Instances'][0]
                                instance_info = requests.get(f"{ORTHANC_URL}/instances/{instance_id}", auth=AUTH).json()
                                
                                # 썸네일 URL
                                thumbnail_url = f"{ORTHANC_URL}/instances/{instance_id}/preview"
                                
                                # 결과에 추가
                                results.append({
                                    "studyID": study_id,
                                    "seriesID": series_id,
                                    "instanceUID": instance_id,
                                    "patientName": patient_name,
                                    "studyDate": study_date,
                                    "modality": modality,
                                    "thumbnail": thumbnail_url,
                                    "sopInstanceUID": instance_info.get('MainDicomTags', {}).get('SOPInstanceUID', '')
                                })
                                
                                # 결과 개수 제한 확인
                                if len(results) >= limit:
                                    break
                    
                    # 결과 개수 제한 확인
                    if len(results) >= limit:
                        break
            
            except Exception as e:
                logger.warning(f"연구 {study_id} 처리 중 오류: {str(e)}")
                continue
        
        return JsonResponse({
            "status": "success",
            "query": query,
            "modality": modality,
            "count": len(results),
            "results": results
        })
    
    except Exception as e:
        logger.exception("인스턴스 검색 실패")
        return JsonResponse({
            "status": "error",
            "message": f"인스턴스 검색 실패: {str(e)}"
        }, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def get_dicom_info(request, instance_id):
    """
    DICOM 인스턴스의 메타데이터 정보 가져오기
    """
    try:
        # 인스턴스 정보 가져오기
        response = requests.get(f"{ORTHANC_URL}/instances/{instance_id}", auth=AUTH)
        if response.status_code != 200:
            return JsonResponse({
                "status": "error",
                "message": f"인스턴스 정보 가져오기 실패: {response.status_code}"
            }, status=500)
        
        instance_info = response.json()
        
        # DICOM 태그 정보 확장
        tags_response = requests.get(f"{ORTHANC_URL}/instances/{instance_id}/tags", auth=AUTH)
        tags = tags_response.json() if tags_response.status_code == 200 else {}
        
        # 필요한 정보 추출
        sop_instance_uid = instance_info.get('MainDicomTags', {}).get('SOPInstanceUID', '')
        patient_info = {
            "patientName": instance_info.get('PatientMainDicomTags', {}).get('PatientName', ''),
            "patientID": instance_info.get('PatientMainDicomTags', {}).get('PatientID', ''),
            "patientBirthDate": instance_info.get('PatientMainDicomTags', {}).get('PatientBirthDate', '')
        }
        
        study_info = {
            "studyDescription": instance_info.get('StudyMainDicomTags', {}).get('StudyDescription', ''),
            "studyDate": instance_info.get('StudyMainDicomTags', {}).get('StudyDate', '')
        }
        
        series_info = {
            "seriesDescription": instance_info.get('SeriesMainDicomTags', {}).get('SeriesDescription', ''),
            "modality": instance_info.get('SeriesMainDicomTags', {}).get('Modality', '')
        }
        
        # 이미지 관련 정보
        image_info = {
            "rows": tags.get('0028,0010', {}).get('Value', ''),
            "columns": tags.get('0028,0011', {}).get('Value', ''),
            "bitsAllocated": tags.get('0028,0100', {}).get('Value', ''),
            "windowCenter": tags.get('0028,1050', {}).get('Value', ''),
            "windowWidth": tags.get('0028,1051', {}).get('Value', '')
        }
        
        return JsonResponse({
            "status": "success",
            "instanceID": instance_id,
            "sopInstanceUID": sop_instance_uid,
            "patientInfo": patient_info,
            "studyInfo": study_info,
            "seriesInfo": series_info,
            "imageInfo": image_info,
            "fullTags": tags
        })
    
    except Exception as e:
        logger.exception("DICOM 정보 가져오기 실패")
        return JsonResponse({
            "status": "error",
            "message": f"DICOM 정보 가져오기 실패: {str(e)}"
        }, status=500)