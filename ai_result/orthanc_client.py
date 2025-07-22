# ai_result/orthanc_client.py

import requests

ORTHANC_URL = 'http://35.225.63.41:8042'  # 실제 서버 주소
AUTH = ('orthanc', 'orthanc')  # 기본 인증 정보

def get_patients():
    """환자 목록"""
    return requests.get(f"{ORTHANC_URL}/patients", auth=AUTH).json()

def get_studies():
    """스터디(검사) 목록"""
    return requests.get(f"{ORTHANC_URL}/studies", auth=AUTH).json()

def get_series():
    """시리즈 목록"""
    return requests.get(f"{ORTHANC_URL}/series", auth=AUTH).json()

def get_instances():
    """인스턴스(DICOM 이미지 단위) 목록"""
    return requests.get(f"{ORTHANC_URL}/instances", auth=AUTH).json()
