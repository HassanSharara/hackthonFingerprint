
import 'package:flutter/material.dart';
import 'package:hweb/Services/Init/init.dart';
import 'package:hweb/ui/launcher/launcher.dart';
import 'package:sharara_apps_building_helpers/ui.dart';

Future<void> main() async {
  await AppInitializer.initApp();
  runApp(ShararaAppHelper(builder: (_)=>const AppLauncher()));
}