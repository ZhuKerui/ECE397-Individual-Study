"""relation_similar_web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path('search/', views.search),
    path('search_related/', views.search_related),
    path('search_relation/', views.search_relation),
    # path('cal_similarity/', views.cal_similarity),
    # path('sent_analysis/', views.sent_analysis),
    # path('get_sent_by_relation/', views.get_sent_by_relation),
    # path('search_relation/', views.search_relation),
    # path('search_similar_pair/', views.find_similar_pairs),
]
