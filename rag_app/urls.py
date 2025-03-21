from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('upload/', views.upload_document, name='upload_document'),
    path('query/', views.query_document, name='query_document'),
]
