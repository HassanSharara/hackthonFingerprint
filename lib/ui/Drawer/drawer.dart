
 import 'package:avatar_glow/avatar_glow.dart';
import 'package:flutter/material.dart';
import 'package:sharara_apps_building_helpers/sharara_apps_building_helpers.dart';
import 'package:sharara_apps_building_helpers/ui.dart';
import 'package:sharara_side_bar/sharara_side_bar.dart';

class MainDrawer extends StatefulWidget {
  const MainDrawer({super.key,
    required this.controller
  });
  final ShararaSideBarController controller;
  @override
  State<MainDrawer> createState() => _MainDrawerState();
}

class _MainDrawerState extends State<MainDrawer> {
  @override
  Widget build(BuildContext context) {
    return  Drawer(
      child:Container(
        decoration:BoxDecoration(
          gradient:LinearGradient(colors: [
             RoyalColors.secondaryColor,
            RoyalColors.mainAppColor
          ]),
          borderRadius:BorderRadius.horizontal(
            left:Radius.circular(60),
          )
        ),
        child:ListView(
          children: [

            SizedBox(
              height:270,
              child: Align(
                alignment:Alignment.center,
                child: AvatarGlow(
                  glowColor:RoyalColors.white.withValues(alpha: 0.7),
                  child: RoyalShadowContainer(
                    shape:BoxShape.circle,
                    backgroundColor:RoyalColors.white,
                    padding:15,
                    child: Icon(Icons.person,size:50,color:RoyalColors.mainAppColor,),
                  ),
                ),
              ),
            ),

            const SizedBox(height:30,),

            item("الحساب الشخصي", Icons.person_2_outlined),
            item("سياسة الخصوصية", Icons.privacy_tip_outlined),
            item("معلومات عنا", Icons.info_outline),
          ],
        ),
      ),
    );
  }

  Widget item(final String title,final IconData iconData,{final GestureTapCallback ?onTap}){
    return GestureDetector(
      onTap: ()async{
        widget.controller.closeDrawer();
        await Future.delayed( Duration(milliseconds:(widget.controller.duration.inMilliseconds * 0.6).toInt() ));
        if(onTap!=null)onTap();
      },
      child: Padding(
        padding:const EdgeInsets.symmetric(horizontal:4,vertical:8),
        child: Row(
          mainAxisAlignment:MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                Icon(iconData,size:28,color:RoyalColors.white,),
                SizedBox(width:9,),
                Text(title,style:TextStyle(
                  color:RoyalColors.white,
                  fontSize:16
                ),),

              ],
            ),
            const Icon(Icons.arrow_forward_ios,size:15,color:RoyalColors.white,)
          ],
        ),
      ),
    );
  }
}
