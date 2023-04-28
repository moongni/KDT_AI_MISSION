from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from .models import Coffee
import json

# Create your views here.

def index(request):
    coffee_all = Coffee.objects.all()
    return render(request, 'index.html', {"coffee_list": coffee_all})


def update_coffee(request, id):
    coffee = get_coffee_by_id(id)
    return render(request, 'update.html', {"coffee": coffee})


def add_coffee(request):
    return render(request, 'add.html')


# CRUD function
class CoffeeCURD(View):

    def post(self, request):
        product = request.POST['product']
        price = request.POST['price']
        code = request.POST['code']

        Coffee.objects.create(product=product, price=price, code=code)
        return HttpResponse("success")


    def put(self, request, id):
        coffee = get_coffee_by_id(id)

        data = json.loads(request.body)
                
        coffee.product = data['product']
        coffee.price = int(data['price'])
        coffee.code = int(data['code'])
        coffee.save()

        return HttpResponse("success")    

    def delete(self, request, id):
        coffee = get_coffee_by_id(id)

        coffee.delete()
        return HttpResponse("success")


def get_coffee_by_id(id):
    return Coffee.objects.get(id=id)