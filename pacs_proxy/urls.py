from django.urls import path, re_path
from pacs_proxy.views import dicom_web_proxy, orthanc_studies, single_study_proxy
from .views import ohif_compatible_study 

urlpatterns = [
    path('studies/', orthanc_studies),

    # /api/dicom-web/ → 빈 path 처리 (OHIF Viewer 최초 요청)
    path('dicom-web/', dicom_web_proxy),
    path('single-study/<str:uid>/', single_study_proxy),
    path('ohif-study/<str:uid>/', ohif_compatible_study, name='ohif_study'),
    # /api/dicom-web/studies/... → 하위 요청 처리
    re_path(r'^dicom-web/(?P<path>.+)$', dicom_web_proxy),
    
]
