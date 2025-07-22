import requests
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import traceback
import json
# 📍 Orthanc 서버 설정
ORTHANC_URL = 'http://localhost:8042'
AUTH = ('orthanc', 'orthanc')  # 기본 인증

# 📍 제외할 헤더 목록
excluded_headers = {
    'host', 'connection', 'keep-alive', 'proxy-authenticate',
    'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
}

# ✅ 1. DICOMweb 프록시
@csrf_exempt
@require_http_methods(["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def dicom_web_proxy(request, path=""):
    method = request.method
    proxied_url = f"{ORTHANC_URL}/dicom-web/{path}".rstrip('/')
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in excluded_headers
    }
    if headers.get('Content-Length', '').strip() == '':
        headers.pop('Content-Length', None)
    if method in ['POST', 'PUT'] and 'Content-Length' not in headers:
        headers['Content-Length'] = str(len(request.body or b''))

    try:
        r = requests.request(
            method=method,
            url=proxied_url,
            headers=headers,
            params=request.GET,
            data=request.body,
            auth=AUTH
        )

        response = HttpResponse(r.content, status=r.status_code)
        for k, v in r.headers.items():
            if k.lower() not in excluded_headers:
                response[k] = v

        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    except Exception as e:
        return JsonResponse({'error': f'Proxy error: {str(e)}'}, status=500)

# ✅ 2. Study 목록 조회
def orthanc_studies(request):
    url = f"{ORTHANC_URL}/dicom-web/studies"
    try:
        response = requests.get(url, headers={'Accept': 'application/json'}, auth=AUTH)
        response.raise_for_status()
        return JsonResponse(response.json(), safe=False)
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)

# ✅ 3. 단일 Study 메타데이터 조회
@csrf_exempt
def single_study_proxy(request, uid):
    url = f"{ORTHANC_URL}/dicom-web/studies/{uid}/metadata"
    try:
        response = requests.get(url, headers={'Accept': 'application/dicom+json'}, auth=AUTH)
        response.raise_for_status()
        return JsonResponse(response.json(), safe=False)
    except Exception as e:
        return JsonResponse({'error': f'Proxy error: {str(e)}'}, status=500)

# ✅ 4. OHIF-compatible JSON 변환
@csrf_exempt
def ohif_compatible_study(request, uid):
    import traceback
    public_dicom_url = "http://35.225.63.41:8000/dicom-web"
    print("🟡 [DEBUG] OHIF 요청 도착 UID:", uid)

    try:
        # 1. Study Metadata 요청
        study_metadata_url = f"{ORTHANC_URL}/dicom-web/studies/{uid}/metadata"
        print("🔵 [DEBUG] Study Metadata 요청:", study_metadata_url)
        study_metadata = requests.get(study_metadata_url, auth=AUTH).json()
        print("🟢 [DEBUG] Study Metadata 수신 완료:", len(study_metadata))
        study = study_metadata[0] if study_metadata else {}

        # 2. Series Metadata 요청
        series_url = f"{ORTHANC_URL}/dicom-web/studies/{uid}/series"
        print("🔵 [DEBUG] Series 요청:", series_url)
        series_metadata = requests.get(series_url, auth=AUTH).json()
        print("🟢 [DEBUG] Series 수신 완료:", len(series_metadata))

        result_series = []
        for series in series_metadata:
            series_uid = series.get("0020000E", {}).get("Value", [None])[0]
            print("🔹 [DEBUG] Series UID:", series_uid)
            if not series_uid:
                continue

            # 3. Instances 요청
            instances_url = f"{ORTHANC_URL}/dicom-web/studies/{uid}/series/{series_uid}/instances"
            print("🔵 [DEBUG] Instances 요청:", instances_url)
            try:
                instances_metadata = requests.get(instances_url, auth=AUTH).json()
                print("🟢 [DEBUG] Instances 수신 완료:", len(instances_metadata))
            except Exception as e:
                print("🔴 [ERROR] 인스턴스 요청 실패:", e)
                traceback.print_exc()
                instances_metadata = []

            instance_list = []
            for instance in instances_metadata:
                sop_uid = instance.get("00080018", {}).get("Value", [None])[0]
                if sop_uid:
                    retrieve_url = f"{public_dicom_url}/studies/{uid}/series/{series_uid}/instances/{sop_uid}/frames/1"
                    instance_list.append({
                        "SOPInstanceUID": sop_uid,
                        "RetrieveURL": retrieve_url,
                        "url": retrieve_url,
                    })

            result_series.append({
                "SeriesInstanceUID": series_uid,
                "SeriesDescription": series.get("0008103E", {}).get("Value", [""])[0],
                "Modality": series.get("00080060", {}).get("Value", [""])[0],
                "Instances": instance_list
            })

        print("✅ [DEBUG] 최종 Series 수:", len(result_series))

        # OHIF가 요구하는 형식의 JSON 생성
        ohif_json = {
            "studies": [
                {
                    "StudyInstanceUID": uid,
                    "StudyDate": study.get("00080020", {}).get("Value", [""])[0],
                    "StudyDescription": study.get("00081030", {}).get("Value", [""])[0],
                    "PatientName": study.get("00100010", {}).get("Value", [{}])[0].get("Alphabetic", ""),
                    "PatientID": study.get("00100020", {}).get("Value", [""])[0],
                    "Series": result_series
                }
            ]
        }

        return HttpResponse(
            json.dumps(ohif_json),
            content_type='application/dicom+json',
            status=200
        )

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': f'OHIF JSON 생성 실패: {str(e)}'}, status=500)