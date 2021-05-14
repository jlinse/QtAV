TEMPLATE = lib # can not create exe for some platforms (winrt, ios). If check header only, staticlib is fine
# not static lib because sometimes we need to check link flags. if qt is static build, this chek may fail. we don't test link for static build because it's impossible to add all dependencies to link flags
INCLUDEPATH += $$[QT_INSTALL_HEADERS]
LIBS += -L$$[QT_INSTALL_LIBS]
CONFIG -= qt app_bundle lib_bundle
CONFIG += console
*msvc*: INCLUDEPATH += $$PWD/../src/compat/msvc
!config_dx: INCLUDEPATH *= $$PWD/../contrib/dxsdk

android {
  message(".PATHS ANDROID QT_ARCH $${QT_ARCH}");
  equals(ANDROID_TARGET_ARCH, armeabi-v7a) {
    message(".PATHS ANDROID FFMPEG armeabi-v7a");
    LIBS += -L$$PWD/../../ffmpeg/android/lib/armeabi-v7a/
    INCLUDEPATH += $$PWD/../../ffmpeg/android/include/
    message("PATHS ANDROID FFMPEG V7A");
    message($$LIBS);
    message($$INCLUDEPATH);
  }
  equals(ANDROID_TARGET_ARCH, arm64-v8a) {
    message(".PATHS ANDROID FFMPEG arm64-v8a");
    LIBS += -L$$PWD/../../ffmpeg/android/lib/arm64-v8a/
    INCLUDEPATH += $$PWD/../../ffmpeg/android/include/
    message(".PATHS ANDROID FFMPEG V8A");
    message($$LIBS);
    message($$INCLUDEPATH);
  }
}
