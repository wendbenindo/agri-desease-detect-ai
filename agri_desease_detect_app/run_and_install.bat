@echo off

REM Nettoyage
echo ========== ETAPE 1 : Nettoyage ==========
call flutter clean

REM Récupération des packages
echo ========== ETAPE 2 : Pub Get ==========
call flutter pub get

REM Compilation APK Debug
echo ========== ETAPE 3 : Build Debug APK ==========
call flutter build apk --debug

REM Installation de l'APK
echo ========== ETAPE 4 : Installation de l APK ==========
call flutter install --use-application-binary=android\app\build\outputs\flutter-apk\app-debug.apk

echo.
echo ========== FIN DU SCRIPT ==========
pause
