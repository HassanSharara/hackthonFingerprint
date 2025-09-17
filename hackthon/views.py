
from django.http import HttpRequest
from django.shortcuts import render
from django.http.response import JsonResponse
from django.core.files.storage import FileSystemStorage
from . import settings
from . import stn_siamese
import os,time
from django.views.decorators.csrf import csrf_exempt
from . import cnn
from . import sift
def showFingerprintView(request:HttpRequest):
    return render(request,"fingerprint/fingerprint-matcher.html")



def findId(name):
   splited = False
   for part in name:
      if splited and part.isdigit():
         return int(part)  
      if part == "_" and not splited:  
         splited = True

@csrf_exempt
def matchFingerprint(request:HttpRequest):
    image = request.FILES.get("image")
    if request.method == "POST" and image:
      location = os.path.join(settings.BASE_DIR,f"hackthon/static/fingerprints/cache/")
      storage = FileSystemStorage(location=location)
      total_file_name = os.path.join(location,image.name)
      storage.save(image.name,image)  
      model = stn_siamese.RectificationTrainer()
      start = time.time()
      best_match,score = model.matchingFingerprint(total_file_name,os.path.join(settings.BASE_DIR,"hackthon/static/hres/easy"))
      end = time.time()
      verified = score > 55
      if score < 8:
         return JsonResponse({
            "status":"err",
            "msg":"the image is not even an fingerprint",
            "verified":verified,
                     "matching-time":f"{end - start} seconds",
         })
      
      if not verified:
         return JsonResponse({
            "status":"err",
            "verified":verified,
                        "msg":"the fingerprint is not verified",
            "matching-time":f"{end - start} seconds",
         })
      return JsonResponse({
         "status":"success",
         "verified":verified,
         "matching-time":f"{end - start} seconds",
         "matching-path":findId(best_match),
         "score":f"{score} %",
        })



@csrf_exempt
def pythonsiamesesift(request:HttpRequest):
    image = request.FILES.get("image")
    if request.method == "POST" and image:
      location = os.path.join(settings.BASE_DIR,f"hackthon/static/fingerprints/cache/")
      storage = FileSystemStorage(location=location)
      total_file_name = os.path.join(location,image.name)
      storage.save(image.name,image)  
      model = sift.SiftMatcherModel()
      start = time.time()
      best_match,score = model.matchingFingerprint(total_file_name,os.path.join(settings.BASE_DIR,"hackthon/static/hres/easy"))
      end = time.time()
      verified = score > 55
      if score < 8:
         return JsonResponse({
            "status":"err",
            "msg":"the image is not even an fingerprint",
            "verified":verified,
                     "matching-time":f"{end - start} seconds",
         })
      
      if not verified:
         return JsonResponse({
            "status":"err",
            "verified":verified,
            "msg":"the fingerprint is not verified",
            "matching-time":f"{end - start} seconds",
         })
      return JsonResponse({
         "status":"success",
         "verified":verified,
         "matching-time":f"{end - start} seconds",
         "matching-path":findId(best_match),
         "score":f"{score} %",
        })

@csrf_exempt
def pythoncnn(request:HttpRequest):
    image = request.FILES.get("image")
    if request.method == "POST" and image:
      location = os.path.join(settings.BASE_DIR,f"hackthon/static/fingerprints/cache/")
      storage = FileSystemStorage(location=location)
      total_file_name = os.path.join(location,image.name)
      storage.save(image.name,image)  
      model = cnn.CnnModel(
         learning_results_path = os.path.join(settings.BASE_DIR,"hackthon/static"),
      )
      start = time.time()
      best_match,score = model.match_fingerprint(total_file_name,os.path.join(settings.BASE_DIR,"hackthon/static/hres/easy"))
      end = time.time()
      return JsonResponse({
         "status":"success",
         "matching-time":f"{end - start} seconds",
         "matching-path":findId(best_match),
         "score":f"{score} %",
      })