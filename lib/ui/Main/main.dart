

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:hweb/AIModel/ai_model.dart';
import 'package:hweb/Constants/Images/images.dart';
import 'package:hweb/ui/Drawer/drawer.dart';
import 'package:image_picker/image_picker.dart';
import 'package:sharara_apps_building_helpers/http.dart';
import 'package:sharara_apps_building_helpers/sharara_apps_building_helpers.dart';
import 'package:sharara_apps_building_helpers/ui.dart';
import 'package:sharara_side_bar/sharara_side_bar.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final ShararaSideBarController sideBarController = ShararaSideBarController();
  File? fingerprintImage;
  AiModel? aiModel;
  dynamic v ;
  @override
  Widget build(BuildContext context) {
    final Size size = MediaQuery.sizeOf(context);
    return ShararaSideBarBuilder(
      sidebar: MainDrawer(
        controller:sideBarController,
      ),
      controller: sideBarController,
      child: Scaffold(
        appBar:AppBar(
          centerTitle:true,
          title:const Text("تطبيق الهاكثون"),
          leading:GestureDetector(
            onTap:()=>sideBarController.openDrawer(),
            child:const Icon(Icons.menu),
          ),
          actions: [
            GestureDetector(
              onTap:()=>FunctionHelpers.jumpTo(context, ShararaThemePicker()),
              child:const Icon(Icons.settings),
            ),
            const SizedBox(width:8,)
          ],
        )
        ,
        body:ListView(
          padding:const EdgeInsets.symmetric(horizontal:5),
          children: [
            SizedBox(height:size.height * 0.10,),
            Center(
              child:Image.asset(ImagesConstants.fingerprint,height:150,),
            ),
      
            const SizedBox(height:30,),
            PopupMenuButton(
                child:RoyalShadowContainer(
                    shadowColor:RoyalColors.secondaryColor.withValues(alpha:0.4),
                    child: Row(
                  mainAxisAlignment:MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      aiModel == null ?
                      "يرجى اختيار نموذج الذكاء الصناعي"
                          : aiModel!.name,
                      style:TextStyle(
                        color:RoyalColors.mainAppColor,
                        fontWeight:FontWeight.w900
                      ),
                    ),

                    const Icon(Icons.arrow_drop_down)
                  ],
                )),
                onSelected: (model){
                  setState(() {
                    aiModel = model;
                  });
                  },
                itemBuilder: (_)=>[
              for(final model in AiModel.models)
                PopupMenuItem(value:model,child: Text(model.name,

                  style:TextStyle(
                      color:RoyalColors.mainAppColor,
                      fontWeight:FontWeight.w900
                  ),
                ),)
            ]),
            const SizedBox(height:30,),
            GestureDetector(
              onTap:() async {
                final file = await FunctionHelpers.tryFuture(ImagePicker().pickImage(source: ImageSource.gallery));
                if( file == null )return;
                setState(() {
                  fingerprintImage = File(file.path);
                });
              },
              child: RoyalShadowContainer(
                child: Column(
                  mainAxisAlignment:MainAxisAlignment.center,
                  crossAxisAlignment:CrossAxisAlignment.center,
                  children: [

                    if (fingerprintImage == null )
                      ...[
                        const Text("يرجى اختيار صورة البصمة",textAlign:TextAlign.center,)
                      ]

                    else ...[
                      Text(fingerprintImage!.path.split("/").last,textAlign:TextAlign.center,)
                    ],
                    const Icon(Icons.add,
                      size:28,
                      color:RoyalColors.greyFaintColor,
                    )
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20,),
      
            RoyalRoundedButton(
              onPressed:_match,
              title:"بدأ التحقق",
            ),
            const SizedBox(height:20,),

            if(v!=null)
              RoyalShadowContainer(
                backgroundColor:RoyalColors.black,
                child:Column(
                  children: [
                    Text(v.toString()
                      .replaceAll("{", "\n{\n")
                      .replaceAll("}", "\n}\n")
                      .replaceAll(",", ",\n"),
                      style:TextStyle(
                        color:
                        v!["status"] == "success" ? RoyalColors.green:
                           RoyalColors.red
                          ,
                        fontWeight:FontWeight.bold
                      ),
                      textDirection:TextDirection.ltr,
                    ),
                    const SizedBox(height:19,),
                    IconButton(onPressed: ()=>setState(() {
                      v = null;
                    }),icon:Icon(Icons.delete,
                      color:RoyalColors.white,
                    ),)
                  ],
                ),
              ),
            const SizedBox(height:50,),
          ],
        ),
      ),
    );
  }

  _match()async{

    if( aiModel == null ) {
      FunctionHelpers.toast("يرجى اختيار نموذج ذكاء صناعي",status:false);
      return;
    }

    final file = fingerprintImage;
    if( file == null ) {
      FunctionHelpers.toast("يرجى اختيار صورة البصمة",status:false);
      return;
    }

    final MultipartFile mFile = await MultipartFile.fromFile(file.path);
    final FormData formData = FormData.fromMap({
      "image":mFile
    });

    final Response? response = await
        ShararaHttp()
    .post(url: aiModel!.url,
         data:formData
       ,withLoading:true );

    if( response == null )return;
    final data = response.data;
    if( data == null) return;
    setState(() {
      v = data;
    });
  }
}
