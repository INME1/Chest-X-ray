from django.urls import path
from . import views
from simclr.api.views import get_first_instance_from_study

urlpatterns = [
    # 이상 탐지 관련 URL 패턴
    path('check-processed/<str:instance_uid>/', views.check_processed_view, name='check_processed'),
    path('process/<str:instance_uid>/', views.process_anomaly_view, name='process_anomaly'),
path('first-instance/<str:study_uid>/', views.get_first_instance_from_study, name='first_instance_from_study'),

    # ✅ 스터디 목록 추가
    path('studies/', views.list_study_metadata, name='list_studies'),
]
