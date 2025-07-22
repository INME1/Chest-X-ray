from django.urls import path, re_path
from . import views
from ai_result import views
from .views import upload_dicom
urlpatterns = [
    # ✅ Orthanc 기본 API
    # path('patients/', views.patient_list_view, name='patient-list'),
    # path('studies/', views.study_list_view, name='study-list'),
    # path('series/', views.series_list_view, name='series-list'),
    # path('instances/', views.instance_list_view, name='instance-list'),

    # ✅ 업로드
    path('upload/', upload_dicom, name='upload_dicom'),

    # ✅ AI 결과
    path('model-results/<str:study_uid>/', views.model_results_view, name='model-results'),

    # ✅ 상세 조회
    path('studies/<str:study_uid>/', views.study_detail_view, name='study-detail'),
    path('series/<str:study_uid>/', views.series_list_by_study, name='series-by-study'),
    path('instances/<str:series_uid>/', views.instance_list_by_series, name='instance-by-series'),
    # ✅ OHIF용 DICOM 프록시
    #re_path(r'^dicom-web/(?P<path>.*)$', dicom_web_proxy, name='dicom-web-proxy'),
]
