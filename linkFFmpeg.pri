isEmpty(FFMPEG_PATH){
  FFMPEG_PATH = ffmpegBin
}
win32:{
  contains(QT_ARCH, i386): {
    INCLUDEPATH += $$PWD/$$FFMPEG_PATH/win/32/include
    LIBS += -L$$PWD/$$FFMPEG_PATH/win/32/lib
  }else{
    INCLUDEPATH += $$PWD/$$FFMPEG_PATH/win/64/include
    LIBS += -L$$PWD/$$FFMPEG_PATH/win/64/lib
  }
}
else:linux:!android:{
  contains(QT_ARCH, i386): {
    INCLUDEPATH += $$PWD/$$FFMPEG_PATH/linux/32/include
    LIBS += -L$$PWD/$$FFMPEG_PATH/linux/32/lib
  }else{
    INCLUDEPATH += $$PWD/$$FFMPEG_PATH/linux/64/include
    LIBS += -L$$PWD/$$FFMPEG_PATH/linux/64/lib
    LIBS += -L$$PWD/$$FFMPEG_PATH/linux/64/lib/ -lavdevice
    LIBS += -L$$PWD/$$FFMPEG_PATH/linux/64/lib/ -lavfilter
    LIBS += -L$$PWD/$$FFMPEG_PATH/linux/64/lib/ -lavformat
    LIBS += -L$$PWD/$$FFMPEG_PATH/linux/64/lib/ -lavcodec
    LIBS += -L$$PWD/$$FFMPEG_PATH/linux/64/lib/ -lavutil
    LIBS += -L$$PWD/$$FFMPEG_PATH/linux/64/lib/ -lswresample
    LIBS += -L$$PWD/$$FFMPEG_PATH/linux/64/lib/ -lswscale
    LIBS += -lvdpau -ldl
  }
}
else:android:{
    INCLUDEPATH += $$PWD/$$FFMPEG_PATH/Android/clang/include

    equals(ANDROID_TARGET_ARCH, armeabi-v7a) {
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/armeabi-v7a/ -lavfilter
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/armeabi-v7a/ -lavformat
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/armeabi-v7a/ -lavcodec
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/armeabi-v7a/ -lavutil
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/armeabi-v7a/ -lswresample
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/armeabi-v7a/ -lswscale
    }
    equals(ANDROID_TARGET_ARCH, arm64-v8a) {
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/arm64-v8a/ -lavfilter
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/arm64-v8a/ -lavformat
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/arm64-v8a/ -lavcodec
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/arm64-v8a/ -lavutil
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/arm64-v8a/ -lswresample
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/arm64-v8a/ -lswscale
    }
    equals(ANDROID_TARGET_ARCH, x86) {
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86/ -lavfilter
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86/ -lavformat
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86/ -lavcodec
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86/ -lavutil
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86/ -lswresample
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86/ -lswscale
    }
    equals(ANDROID_TARGET_ARCH, x86_64) {
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86_64/ -lavfilter
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86_64/ -lavformat
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86_64/ -lavcodec
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86_64/ -lavutil
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86_64/ -lswresample
        LIBS += -L$$PWD/$$FFMPEG_PATH/Android/clang/lib/x86_64/ -lswscale
    }

}
else:macx:{
    INCLUDEPATH += $$PWD/$$FFMPEG_PATH/linux/64/include
    LIBS += -L$$PWD/$$FFMPEG_PATH/mac/lib
    LIBS += -L$$PWD/$$FFMPEG_PATH/mac/lib/ -lavdevice
    LIBS += -L$$PWD/$$FFMPEG_PATH/mac/lib/ -lavfilter
    LIBS += -L$$PWD/$$FFMPEG_PATH/mac/lib/ -lavformat
    LIBS += -L$$PWD/$$FFMPEG_PATH/mac/lib/ -lavcodec
    LIBS += -L$$PWD/$$FFMPEG_PATH/mac/lib/ -lavutil
    LIBS += -L$$PWD/$$FFMPEG_PATH/mac/lib/ -lswresample
    LIBS += -L$$PWD/$$FFMPEG_PATH/mac/lib/ -lswscale
}
else:mac:{ #ios
    INCLUDEPATH += $$PWD/$$FFMPEG_PATH/linux/64/include
    LIBS += -L$$PWD/$$FFMPEG_PATH/ios/lib/ -lavfilter
    LIBS += -L$$PWD/$$FFMPEG_PATH/ios/lib/ -lavformat
    LIBS += -L$$PWD/$$FFMPEG_PATH/ios/lib/ -lavcodec
    LIBS += -L$$PWD/$$FFMPEG_PATH/ios/lib/ -lavutil
    LIBS += -L$$PWD/$$FFMPEG_PATH/ios/lib/ -lswresample
    LIBS += -L$$PWD/$$FFMPEG_PATH/ios/lib/ -lswscale
}
message("FFMPEG_PATH = " $$FFMPEG_PATH);
message("INCLUDEPATH = " $$INCLUDEPATH);
message("LIBS = " $$LIBS);
