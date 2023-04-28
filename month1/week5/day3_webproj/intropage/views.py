from django.shortcuts import render
from django.views import View

# Create your views here.

class IntroView(View):
    
    def get(self, request, *args, **kwargs):
        info = {
            'name': '문건희',
            'age': 27,
            'mbti': 'intp',
            'height': '182.2 cm',
            'hobby': ['보디빌딩', '서핑'],
            'stack': ['Python', 'Javascript', 'C++/C', 'MySQL', 'MongoDB'],
            'description':"""
안녕하세요. 문건희입니다. 
프로그래머스 인공지능 데브코스 5기를 수강하고 있으며 향후 AI 모델 개발 및 모델 서빙 백엔드 개발 방향으로 취업하기를 희망하고 있습니다.
            """
        }
        return render(request, 'intro.html', {'info': info})