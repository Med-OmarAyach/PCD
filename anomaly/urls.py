from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.signin_view, name='sign_in'),
    path('api/get_room_data/', views.get_room_data, name='get_room_data'),
    path('api/get_room_history/', views.get_room_history, name='get_room_history'),
    path('api/test/', views.test_api_view, name='test_api'),
    path('loggedin/', views.main_page_view, name='main_page'),
    path('signup/', views.signup_view, name='sign_up'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)