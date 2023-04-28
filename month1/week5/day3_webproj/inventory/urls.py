from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('update.html/<int:id>', views.update_coffee),
    path('add.html', views.add_coffee),
    path('coffees', views.CoffeeCURD.as_view()),
    path('coffess/<int:id>', views.CoffeeCURD.as_view()),
]