from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from pacs_proxy.views import dicom_web_proxy
from supervised_learning import views
from pacs_proxy.views import orthanc_studies
from simclr.api import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # 각 앱 URL
    path('api/', include('pacs_proxy.urls')),
    path('api/', include('ai_result.urls')),
    path('api/', include('supervised_learning.urls')), 
    path('api/anomaly/', include('simclr.api.urls')),
    path('api/studies/', orthanc_studies), 
    # Orthanc DICOMweb 프록시
    path('dicom-web/<path:path>', dicom_web_proxy),
    path('api/anomaly/first-instance/<str:study_uid>/', views.get_first_instance_from_study),
    path('api/anomaly/process/<str:instance_uid>/', views.process_anomaly_view),
] 
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
