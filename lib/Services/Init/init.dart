

 import 'package:hive_flutter/hive_flutter.dart';
import 'package:sharara_apps_building_helpers/ui.dart';
final class AppInitializer {
   const AppInitializer._();
   static Future<void> initApp()async{
     await _initAppsBuildingHelper();
   }

   static Future<void> _initAppsBuildingHelper()async{
     await Hive.initFlutter();
     await ShararaAppHelperInitializer
         .initialize(
       withHiveFlutter:true,
     );
   }
 }