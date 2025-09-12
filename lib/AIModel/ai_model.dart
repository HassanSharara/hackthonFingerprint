import 'package:hweb/Constants/Api/api.dart';

 final class AiModel {
   final String url,name;
   const AiModel({
     required this.url,
     required this.name,
 });


   static const List<AiModel> models = [
     AiModel(url: ApiConstants.cnn, name: "python cnn (the fastest when its comes to OF)"),
     AiModel(url: ApiConstants.pythonStn, name: "python stn siamese ai model"),
     AiModel(url: ApiConstants.pythonSift, name: "python stnSift"),
     AiModel(url: ApiConstants.matchFingerprint, name: "Rust stn siamese ai model (takes time because is not trained enough)"),
     AiModel(url: ApiConstants.sift, name: "Rust sift (flan and sift mixed algorithm)"),

   ];
 }