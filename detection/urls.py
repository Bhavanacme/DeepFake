from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('contact/', views.contact_view, name='contact'),
    path('how-it-works/', views.how_it_works_view, name='how_it_works'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('detect/', views.detect_upload_view, name='detect'),
    path('result/<int:pk>/', views.result_view, name='detection_result'),
    path('history/', views.history_view, name='history'),
    path('history/<int:pk>/', views.details_view, name='detection_details'),
    path('history/<int:pk>/delete/', views.delete_result_view, name='delete_result'),
    path('report/<int:pk>/', views.report_view, name='report'),
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
]
