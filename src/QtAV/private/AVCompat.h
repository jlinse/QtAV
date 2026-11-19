/******************************************************************************
    QtAV:  Multimedia framework based on Qt and FFmpeg
	solve the version problem and diffirent api in FFmpeg and libav
    Copyright (C) 2012-2018 Wang Bin <wbsecg1@gmail.com>

*   This file is part of QtAV

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
******************************************************************************/
#ifndef QTAV_COMPAT_H
#define QTAV_COMPAT_H

/*!
  NOTE: include this at last
 */
#define QTAV_USE_FFMPEG(MODULE) (MODULE##_VERSION_MICRO >= 100)
#define QTAV_USE_LIBAV(MODULE)  !QTAV_USE_FFMPEG(MODULE)
#define FFMPEG_MODULE_CHECK(MODULE, MAJOR, MINOR, MICRO) \
    (QTAV_USE_FFMPEG(MODULE) && MODULE##_VERSION_INT >= AV_VERSION_INT(MAJOR, MINOR, MICRO))
#define LIBAV_MODULE_CHECK(MODULE, MAJOR, MINOR, MICRO) \
    (QTAV_USE_LIBAV(MODULE) && MODULE##_VERSION_INT >= AV_VERSION_INT(MAJOR, MINOR, MICRO))
#define AV_MODULE_CHECK(MODULE, MAJOR, MINOR, MICRO, MINOR2, MICRO2) \
    (LIBAV_MODULE_CHECK(MODULE, MAJOR, MINOR, MICRO) || FFMPEG_MODULE_CHECK(MODULE, MAJOR, MINOR2, MICRO2))
/// example: AV_ENSURE(avcodec_close(avctx), false) will print error and return false if failed. AV_WARN just prints error.
#define AV_ENSURE_OK(FUNC, ...) AV_RUN_CHECK(FUNC, return, __VA_ARGS__)
#define AV_ENSURE(FUNC, ...) AV_RUN_CHECK(FUNC, return, __VA_ARGS__)
#define AV_WARN(FUNC) AV_RUN_CHECK(FUNC, void)

#include "QtAV/QtAV_Global.h"
#ifdef __cplusplus
extern "C"
{
/*UINT64_C: C99 math features, need -D__STDC_CONSTANT_MACROS in CXXFLAGS*/
#endif /*__cplusplus*/
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
#if LIBAVCODEC_VERSION_MAJOR >= 58
#include <libavcodec/bsf.h>
#endif
#include <libavutil/avutil.h>
#include <libavutil/avstring.h>
#include <libavutil/dict.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/mathematics.h> //AV_ROUND_UP, av_rescale_rnd for libav
#include <libavutil/cpu.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libavutil/parseutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/avstring.h>
#include <libavfilter/version.h>

#define AVCODEC_STATIC_REGISTER FFMPEG_MODULE_CHECK(LIBAVCODEC, 58, 10, 100)
#define AVFORMAT_STATIC_REGISTER FFMPEG_MODULE_CHECK(LIBAVFORMAT, 58, 9, 100)
#define AVFILTER_STATIC_REGISTER FFMPEG_MODULE_CHECK(LIBAVFILTER, 7, 14, 100)

#if !FFMPEG_MODULE_CHECK(LIBAVUTIL, 51, 73, 101)
#include <libavutil/channel_layout.h>
#endif

/* TODO: how to check whether we have swresample or not? how to check avresample?*/
#include <libavutil/samplefmt.h>
#if QTAV_HAVE(SWRESAMPLE)
#include <libswresample/swresample.h>
#ifndef LIBSWRESAMPLE_VERSION_INT //ffmpeg 0.9, swr 0.5
#define LIBSWRESAMPLE_VERSION_INT AV_VERSION_INT(LIBSWRESAMPLE_VERSION_MAJOR, LIBSWRESAMPLE_VERSION_MINOR, LIBSWRESAMPLE_VERSION_MICRO)
#endif //LIBSWRESAMPLE_VERSION_INT
//ffmpeg >= 0.11.x. swr0.6.100: ffmpeg-0.10.x
#define HAVE_SWR_GET_DELAY (LIBSWRESAMPLE_VERSION_INT > AV_VERSION_INT(0, 6, 100))
#endif //QTAV_HAVE(SWRESAMPLE)
#if QTAV_HAVE(AVRESAMPLE)
#include <libavresample/avresample.h>
#endif //QTAV_HAVE(AVRESAMPLE)

#if QTAV_HAVE(AVFILTER)
#if LIBAVFILTER_VERSION_INT < AV_VERSION_INT(3,8,0)
#include <libavfilter/avfiltergraph.h> /*code is here for old version*/
#else
#include <libavfilter/avfilter.h>
#endif
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#endif //QTAV_HAVE(AVFILTER)

#if QTAV_HAVE(AVDEVICE)
#include <libavdevice/avdevice.h>
#endif

#ifdef __cplusplus
}
#endif /*__cplusplus*/

/*!
 * Guide to uniform the api for different FFmpeg version(or other libraries)
 * We use the existing old api to simulater .
 * 1. The old version does not have this api: Just add it.
 * 2. The old version has similar api: Try using macro.
 * e.g. the old is bool my_play(char* data, size_t size)
 *      the new is bool my_play2(const ByteArray& data)
 * change:
 *    #define my_play2(data) my_play(data.data(), data.size());
 *
 * 3. The old version api is conflicted with the latest's. We can redefine the api
 * e.g. the old is bool my_play(char* data, size_t size)
 *      the new is bool my_play(const ByteArray& data)
 * change:
 *    typedef bool (*my_play_t)(const ByteArray&);
 *    static my_play_t my_play_ptr = my_play; //using the existing my_play(char*, size_t)
 *    #define my_play my_play_compat
 *    inline bool my_play_compat(const ByteArray& data)
 *    {
 *        return my_play_ptr(data.data(), data.size());
 *    }
 * 4. conflict macros
 * see av_err2str
 */

#ifndef AV_VERSION_INT
#define AV_VERSION_INT(a, b, c) (a<<16 | b<<8 | c)
#endif /*AV_VERSION_INT*/

void ffmpeg_version_print();

#if !FFMPEG_MODULE_CHECK(LIBAVFORMAT, 56, 4, 101)
int avio_feof(AVIOContext *s);
#endif
#if QTAV_USE_LIBAV(LIBAVFORMAT)
int avformat_alloc_output_context2(AVFormatContext **avctx, AVOutputFormat *oformat, const char *format, const char *filename);
#endif
//TODO: always inline
/* --gnu option of the RVCT compiler also defines __GNUC__ */
#if defined(__GNUC__) && !(defined(__ARMCC__) || defined(__CC_ARM))
#define GCC_VERSION_AT_LEAST(major, minor, patch) \
    (__GNUC__ > major || (__GNUC__ == major && (__GNUC_MINOR__ > minor \
    || (__GNUC_MINOR__ == minor && __GNUC_PATCHLEVEL__ >= patch))))
#else
/* Define this for !GCC compilers, just so we can write things like GCC_VERSION_AT_LEAST(4, 1, 0). */
#define GCC_VERSION_AT_LEAST(major, minor, patch) 0
#endif

//FFmpeg2.0, Libav10 2013-03-08 - Reference counted buffers - lavu 52.19.100/52.8.0, lavc 55.0.100 / 55.0.0, lavf 55.0.100 / 55.0.0, lavd 54.4.100 / 54.0.0, lavfi 3.5.0
#define QTAV_HAVE_AVBUFREF AV_MODULE_CHECK(LIBAVUTIL, 52, 8, 0, 19, 100)

#if defined(_MSC_VER) || !defined(av_err2str) || (GCC_VERSION_AT_LEAST(4, 7, 0) && __cplusplus)
#ifdef av_err2str
#undef av_err2str
/*#define av_make_error_string qtav_make_error_string*/
#else
/**
 * Fill the provided buffer with a string containing an error string
 * corresponding to the AVERROR code errnum.
 *
 * @param errbuf         a buffer
 * @param errbuf_size    size in bytes of errbuf
 * @param errnum         error code to describe
 * @return the buffer in input, filled with the error description
 * @see av_strerror()
 */
static av_always_inline char *av_make_error_string(char *errbuf, size_t errbuf_size, int errnum)
{
	av_strerror(errnum, errbuf, errbuf_size);
	return errbuf;
}
#endif /*av_err2str*/

#define AV_ERROR_MAX_STRING_SIZE 64
#ifdef QT_CORE_LIB
#include <QtCore/QSharedPointer>
#define av_err2str(e) av_err2str_qsp(e).data()
av_always_inline QSharedPointer<char> av_err2str_qsp(int errnum)
{
    QSharedPointer<char> str((char*)calloc(AV_ERROR_MAX_STRING_SIZE, 1), ::free);
    av_strerror(errnum, str.data(), AV_ERROR_MAX_STRING_SIZE);
    return str;
}
#else
av_always_inline char* av_err2str(int errnum)
{
    static char str[AV_ERROR_MAX_STRING_SIZE];
    memset(str, 0, sizeof(str));
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
#endif /* QT_CORE_LIB */
#endif /*!defined(av_err2str) || GCC_VERSION_AT_LEAST(4, 7, 2)*/

#if (LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(52,23,0))
#define avcodec_decode_audio3(avctx, samples, frame_size_ptr, avpkt) \
    avcodec_decode_audio2(avctx, samples, frame_size_ptr, (*avpkt).data, (*avpkt).size);

#endif /*AV_VERSION_INT(52,23,0)*/

#if (LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(52,101,0))
#define av_dump_format(...) dump_format(__VA_ARGS__)
#endif /*AV_VERSION_INT(52,101,0)*/

#if QTAV_HAVE(SWRESAMPLE) && (LIBSWRESAMPLE_VERSION_INT <= AV_VERSION_INT(0, 5, 0))
#define swresample_version() LIBSWRESAMPLE_VERSION_INT //we can not know the runtime version, so just use build time version
#define swresample_configuration() "Not available."
#define swresample_license() "Not available."
#endif

#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(51, 32, 0)
int64_t av_get_default_channel_layout(int nb_channels);
#endif
/*
 * mapping avresample to swresample
 * https://github.com/xbmc/xbmc/commit/274679d
 */
#if (QTAV_HAVE(SWR_AVR_MAP) || !QTAV_HAVE(SWRESAMPLE)) && QTAV_HAVE(AVRESAMPLE)
#ifndef SWR_CH_MAX
#ifdef AVRESAMPLE_MAX_CHANNELS
#define SWR_CH_MAX AVRESAMPLE_MAX_CHANNELS
#else
#define SWR_CH_MAX 64
#endif //AVRESAMPLE_MAX_CHANNELS
#endif //SWR_CH_MAX
#define SwrContext AVAudioResampleContext
#define swr_init(ctx) avresample_open(ctx)
//free context and set pointer to null. see swresample
#define swr_free(ctx) \
    if (ctx && *ctx) { \
        avresample_close(*ctx); \
        *ctx = 0; \
    }
#define swr_get_class() avresample_get_class()
#define swr_alloc() avresample_alloc_context()
//#define swr_next_pts()
#define swr_set_compensation() avresample_set_compensation()
#define swr_set_channel_mapping(ctx, map) avresample_set_channel_mapping(ctx, map)
#define swr_set_matrix(ctx, matrix, stride) avresample_set_matrix(ctx, matrix, stride)
//#define swr_drop_output(ctx, count)
//#define swr_inject_silence(ctx, count)
#define swr_get_delay(ctx, ...) avresample_get_delay(ctx)
#if LIBAVRESAMPLE_VERSION_INT >= AV_VERSION_INT(1, 0, 0) //ffmpeg >= 1.1
#define swr_convert(ctx, out, out_count, in, in_count) \
    avresample_convert(ctx, out, 0, out_count, const_cast<uint8_t**>(in), 0, in_count)
#else
#define swr_convert(ctx, out, out_count, in, in_count) \
    avresample_convert(ctx, (void**)out, 0, out_count, (void**)in, 0, in_count)
#define HAVE_SWR_GET_DELAY 1
#define swr_get_delay(ctx, ...) avresample_get_delay(ctx)
#endif
struct SwrContext *swr_alloc_set_opts(struct SwrContext *s, int64_t out_ch_layout, enum AVSampleFormat out_sample_fmt, int out_sample_rate, int64_t in_ch_layout, enum AVSampleFormat in_sample_fmt, int in_sample_rate, int log_offset, void *log_ctx);
#define swresample_version() avresample_version()
#define swresample_configuration() avresample_configuration()
#define swresample_license() avresample_license()
#endif //MAP_SWR_AVR


/* For FFmpeg < 2.0
 * FF_API_PIX_FMT macro?
 * 51.42.0: PIX_FMT_* -> AV_PIX_FMT_*, PixelFormat -> AVPixelFormat
 * so I introduce QTAV_PIX_FMT_C(X) for internal use
 * FFmpeg n1.1 AVPixelFormat
 */
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 13, 100) //(51, 42, 0)
typedef enum PixelFormat AVPixelFormat; // so we must avoid using  enum AVPixelFormat
#define QTAV_PIX_FMT_C(X) PIX_FMT_##X
#else //FFmpeg >= 2.0
typedef enum AVPixelFormat AVPixelFormat;
#define QTAV_PIX_FMT_C(X) AV_PIX_FMT_##X
#endif //AV_VERSION_INT(51, 42, 0)
// FF_API_PIX_FMT
#ifdef PixelFormat
#undef PixelFormat
#endif

// AV_PIX_FMT_FLAG_XXX was PIX_FMT_XXX before FFmpeg 2.0
// AV_PIX_FMT_FLAG_ALPHA was added at 52.2.0. but version.h not changed
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 2, 1) //git cbe5a60c9d495df0fb4775b064f06719b70b9952
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(51, 22, 1) //git 38d553322891c8e47182f05199d19888422167dc
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(51, 19, 0) //git 6b0768e2021b90215a2ab55ed427bce91d148148
#define PIX_FMT_PLANAR   16 ///< At least one pixel component is not in the first data plane
#define PIX_FMT_RGB      32 ///< The pixel format contains RGB-like data (as opposed to YUV/grayscale)
#endif //AV_VERSION_INT(51, 19, 0)
#define PIX_FMT_PSEUDOPAL 64  //why not defined in FFmpeg 0.9 lavu51.32.0 but git log says 51.22.1 defined it?
#endif //AV_VERSION_INT(51, 22, 1)
#define PIX_FMT_ALPHA   128 ///< The pixel format has an alpha channel
#endif //AV_VERSION_INT(52, 2, 1)

#ifndef PIX_FMT_PLANAR
#define PIX_FMT_PLANAR 16
#endif //PIX_FMT_PLANAR
#ifndef PIX_FMT_RGB
#define PIX_FMT_RGB 32
#endif //PIX_FMT_RGB
#ifndef PIX_FMT_PSEUDOPAL
#define PIX_FMT_PSEUDOPAL 64
#endif //PIX_FMT_PSEUDOPAL
#ifndef PIX_FMT_ALPHA
#define PIX_FMT_ALPHA 128
#endif //PIX_FMT_ALPHA

/*
 * rename PIX_FMT_* flags to AV_PIX_FMT_FLAG_*. git e6c4ac7b5f038be56dfbb0171f5dd0cb850d9b28
 */
//#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 11, 0)
#ifndef AV_PIX_FMT_FLAG_BE
#define AV_PIX_FMT_FLAG_BE           PIX_FMT_BE
#define AV_PIX_FMT_FLAG_PAL          PIX_FMT_PAL
#define AV_PIX_FMT_FLAG_BITSTREAM    PIX_FMT_BITSTREAM
#define AV_PIX_FMT_FLAG_HWACCEL      PIX_FMT_HWACCEL

// FFmpeg >=  0.9, libav >= 0.8.8(51,22,1)
#define AV_PIX_FMT_FLAG_PLANAR       PIX_FMT_PLANAR
#define AV_PIX_FMT_FLAG_RGB          PIX_FMT_RGB

// FFmpeg >= 1.0, libav >= 9.7
#define AV_PIX_FMT_FLAG_PSEUDOPAL    PIX_FMT_PSEUDOPAL
// FFmpeg >= 1.1, libav >= 9.7
#define AV_PIX_FMT_FLAG_ALPHA        PIX_FMT_ALPHA
#endif //AV_PIX_FMT_FLAG_BE
//#endif //AV_VERSION_INT(52, 11, 0)
// FFmpeg >= 1.1, but use internal av_pix_fmt_descriptors. FFmpeg < 1.1 has extern av_pix_fmt_descriptors
// used by av_pix_fmt_count_planes
#if !AV_MODULE_CHECK(LIBAVUTIL, 52, 3, 0, 13, 100)
const AVPixFmtDescriptor *av_pix_fmt_desc_get(AVPixelFormat pix_fmt);
const AVPixFmtDescriptor *av_pix_fmt_desc_next(const AVPixFmtDescriptor *prev);
AVPixelFormat av_pix_fmt_desc_get_id(const AVPixFmtDescriptor *desc);
#endif // !AV_MODULE_CHECK(LIBAVUTIL, 52, 3, 0, 13, 100)
#if !FFMPEG_MODULE_CHECK(LIBAVUTIL, 52, 48, 101) // since ffmpeg2.1, libavutil53.16.0 (FF_API_AVFRAME_COLORSPACE), git 8c02adc
enum AVColorSpace av_frame_get_colorspace(const AVFrame *frame);
enum AVColorRange av_frame_get_color_range(const AVFrame *frame);
#endif
/*
 * lavu 52.9.0 git 2c328a907978b61949fd20f7c991803174337855
 * FFmpeg >= 2.0.
 */
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 38, 100)
int av_pix_fmt_count_planes(AVPixelFormat pix_fmt);
#endif //AV_VERSION_INT(52, 38, 100)

// FFmpeg < 1.0 has no av_samples_copy
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(51, 73, 101)
/**
 * Copy samples from src to dst.
 *
 * @param dst destination array of pointers to data planes
 * @param src source array of pointers to data planes
 * @param dst_offset offset in samples at which the data will be written to dst
 * @param src_offset offset in samples at which the data will be read from src
 * @param nb_samples number of samples to be copied
 * @param nb_channels number of audio channels
 * @param sample_fmt audio sample format
 */
int av_samples_copy(uint8_t **dst, uint8_t * const *src, int dst_offset,
                    int src_offset, int nb_samples, int nb_channels,
                    enum AVSampleFormat sample_fmt);
#endif //AV_VERSION_INT(51, 73, 101)

// < ffmpeg 1.0
//#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(54, 59, 100)
#if AV_MODULE_CHECK(LIBAVCODEC, 54, 25, 0, 51, 100)
#define QTAV_CODEC_ID(X) AV_CODEC_ID_##X
#else
typedef enum CodecID AVCodecID;
#define QTAV_CODEC_ID(X) CODEC_ID_##X
#endif

/* av_frame_alloc
 * since FFmpeg2.0: 2.0.4 avcodec-55.18.102, avutil-52.38.100 (1.2.7 avcodec-54.92.100,avutil-52.18.100)
 * since libav10.0: 10.2 avcodec55.34.1, avutil-53.3.0
 * the same as avcodec_alloc_frame() (deprecated since 2.2). AVFrame was in avcodec.h, now in avutil/frame.h
 */
#if !AV_MODULE_CHECK(LIBAVCODEC, 55, 34, 0, 18, 100)
#define av_frame_alloc() avcodec_alloc_frame()
#if QTAV_USE_LIBAV(LIBAVCODEC) || FFMPEG_MODULE_CHECK(LIBAVCODEC, 54, 59, 100)
#define av_frame_free(f) avcodec_free_frame(f)
#else
#define av_frame_free(f) av_free(f)
#endif
#endif

#if QTAV_USE_LIBAV(LIBAVCODEC)
const char *avcodec_get_name(enum AVCodecID id);
#endif
#if !AV_MODULE_CHECK(LIBAVCODEC, 55, 55, 0, 68, 100)
void av_packet_rescale_ts(AVPacket *pkt, AVRational src_tb, AVRational dst_tb);
#endif
// since libav-11, ffmpeg-2.1
#if !LIBAV_MODULE_CHECK(LIBAVCODEC, 56, 1, 0) && !FFMPEG_MODULE_CHECK(LIBAVCODEC, 55, 39, 100)
int av_packet_copy_props(AVPacket *dst, const AVPacket *src);
#endif
// since libav-10, ffmpeg-2.1
#if !LIBAV_MODULE_CHECK(LIBAVCODEC, 55, 34, 1) && !FFMPEG_MODULE_CHECK(LIBAVCODEC, 55, 39, 100)
void av_packet_free_side_data(AVPacket *pkt);
#endif
//ffmpeg2.1 libav10
#if !AV_MODULE_CHECK(LIBAVCODEC, 55, 34, 1, 39, 101)
int av_packet_ref(AVPacket *dst, const AVPacket *src);
#define av_packet_unref(pkt) av_free_packet(pkt)
#endif

#if !AV_MODULE_CHECK(LIBAVCODEC, 55, 52, 0, 63, 100)
void avcodec_free_context(AVCodecContext **pavctx);
#endif

#if QTAV_HAVE(AVFILTER)
// ffmpeg2.0 2013-07-03 - 838bd73 - lavfi 3.78.100 - avfilter.h
#if QTAV_USE_LIBAV(LIBAVFILTER)
#define avfilter_graph_parse_ptr(pGraph, pFilters, ppInputs, ppOutputs, pLog) avfilter_graph_parse(pGraph, pFilters, *ppInputs, *ppOutputs, pLog)
#elif !FFMPEG_MODULE_CHECK(LIBAVFILTER, 3, 78, 100)
#define avfilter_graph_parse_ptr(pGraph, pFilters, ppInputs, ppOutputs, pLog) avfilter_graph_parse(pGraph, pFilters, ppInputs, ppOutputs, pLog)
#endif //QTAV_USE_LIBAV(LIBAVFILTER)

//ffmpeg1.0 2012-06-12 - c7b9eab / 84b9fbe - lavfi 2.79.100 / 2.22.0 - avfilter.h
#if !AV_MODULE_CHECK(LIBAVFILTER, 2, 22, 0, 79, 100) //FF_API_AVFILTERPAD_PUBLIC
const char *avfilter_pad_get_name(const AVFilterPad *pads, int pad_idx);
enum AVMediaType avfilter_pad_get_type(const AVFilterPad *pads, int pad_idx);
#endif
///ffmpeg1.0 lavfi 2.74.100 / 2.17.0. was in ffmpeg <libavfilter/avcodec.h> in old ffmpeg and now are in avfilter.h and deprecated. declare here to avoid version check
#if QTAV_USE_FFMPEG(LIBAVFILTER)
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
struct AVFilterBufferRef;
int avfilter_copy_buf_props(AVFrame *dst, const AVFilterBufferRef *src);
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif
#endif //QTAV_HAVE(AVFILTER)

/* helper functions */
const char *get_codec_long_name(AVCodecID id);

// lavf 59 (FFmpeg 5.0) made AVInputFormat and AVOutputFormat pointers const
#if LIBAVFORMAT_VERSION_MAJOR >= 59
typedef const AVInputFormat QtAVInputFormat;
typedef const AVOutputFormat QtAVOutputFormat;
#else
typedef AVInputFormat QtAVInputFormat;
typedef AVOutputFormat QtAVOutputFormat;
#endif

// AVStream.codec, avcodec_copy_context() and avcodec_close() are removed. avcodec_free_context() closes the context
#if LIBAVCODEC_VERSION_MAJOR >= 61
#define QTAV_NO_AVCODEC_COPY_CONTEXT 1
#endif
#if LIBAVCODEC_VERSION_MAJOR >= 62
#define QTAV_NO_AVCODEC_CLOSE 1
#endif
/// copy codec parameters (including extradata and time base) from src to dst. dst must not be open.
int qtav_copy_codec_context(AVCodecContext *dst, const AVCodecContext *src);
#ifdef QTAV_NO_AVCODEC_COPY_CONTEXT
#define avcodec_copy_context qtav_copy_codec_context
#endif
/// avcodec_close() was removed. *ctx is replaced by a new context with the same parameters if it can't be closed, so it can be opened again
int qtav_close_codec_context(AVCodecContext **ctx);
/// alloc a context filled with the parameters from stream. use avcodec_free_context() to free it
AVCodecContext* qtav_alloc_stream_codec_context(const AVStream *stream);

// AVFrame accessors were removed (FFmpeg 6.0), use the fields. AVFrame.pkt_pts is removed too, AVFrame.pts has the packet pts
#if LIBAVUTIL_VERSION_MAJOR >= 57
#define av_frame_get_colorspace(f) ((f)->colorspace)
#define av_frame_get_color_range(f) ((f)->color_range)
#endif
#if LIBAVUTIL_VERSION_MAJOR >= 58
#define QTAV_FRAME_PKT_PTS(f) ((f)->pts)
#else
#define QTAV_FRAME_PKT_PTS(f) ((f)->pkt_pts)
#endif

// AVCodec.pix_fmts, sample_fmts, supported_samplerates and channel_layouts are removed. they are NULL if everything is supported
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 13, 100)
static inline const void* qtav_codec_config(const AVCodec *c, enum AVCodecConfig type)
{
    const void *v = NULL;
    if (avcodec_get_supported_config(NULL, c, type, 0, &v, NULL) < 0)
        return NULL;
    return v;
}
static inline const AVPixelFormat* qtav_codec_pix_fmts(const AVCodec *c)
{ return (const AVPixelFormat*)qtav_codec_config(c, AV_CODEC_CONFIG_PIX_FORMAT); }
static inline const AVSampleFormat* qtav_codec_sample_fmts(const AVCodec *c)
{ return (const AVSampleFormat*)qtav_codec_config(c, AV_CODEC_CONFIG_SAMPLE_FORMAT); }
static inline const int* qtav_codec_samplerates(const AVCodec *c)
{ return (const int*)qtav_codec_config(c, AV_CODEC_CONFIG_SAMPLE_RATE); }
/// the first supported layout mask, 0 if all or not a native layout
static inline int64_t qtav_codec_first_channel_layout(const AVCodec *c)
{
    const AVChannelLayout *cl = (const AVChannelLayout*)qtav_codec_config(c, AV_CODEC_CONFIG_CHANNEL_LAYOUT);
    return cl && cl->nb_channels > 0 && cl->order == AV_CHANNEL_ORDER_NATIVE ? (int64_t)cl->u.mask : 0;
}
#else
static inline const AVPixelFormat* qtav_codec_pix_fmts(const AVCodec *c) { return c->pix_fmts; }
static inline const AVSampleFormat* qtav_codec_sample_fmts(const AVCodec *c) { return c->sample_fmts; }
static inline const int* qtav_codec_samplerates(const AVCodec *c) { return c->supported_samplerates; }
static inline int64_t qtav_codec_first_channel_layout(const AVCodec *c)
{ return c->channel_layouts ? (int64_t)c->channel_layouts[0] : 0; }
#endif

// FFmpeg 6.0 (lavc 60) renamed FF_PROFILE_* to AV_PROFILE_*, and the old names are removed in FFmpeg 8.0
#ifndef FF_PROFILE_H264_BASELINE
#ifdef AV_PROFILE_H264_BASELINE
#define FF_PROFILE_UNKNOWN AV_PROFILE_UNKNOWN
#define FF_PROFILE_MPEG2_SIMPLE AV_PROFILE_MPEG2_SIMPLE
#define FF_PROFILE_MPEG2_MAIN AV_PROFILE_MPEG2_MAIN
#define FF_PROFILE_H264_BASELINE AV_PROFILE_H264_BASELINE
#define FF_PROFILE_H264_CONSTRAINED_BASELINE AV_PROFILE_H264_CONSTRAINED_BASELINE
#define FF_PROFILE_H264_MAIN AV_PROFILE_H264_MAIN
#define FF_PROFILE_H264_HIGH AV_PROFILE_H264_HIGH
#define FF_PROFILE_MPEG4_SIMPLE AV_PROFILE_MPEG4_SIMPLE
#define FF_PROFILE_MPEG4_MAIN AV_PROFILE_MPEG4_MAIN
#define FF_PROFILE_MPEG4_ADVANCED_SIMPLE AV_PROFILE_MPEG4_ADVANCED_SIMPLE
#define FF_PROFILE_VC1_SIMPLE AV_PROFILE_VC1_SIMPLE
#define FF_PROFILE_VC1_MAIN AV_PROFILE_VC1_MAIN
#define FF_PROFILE_VC1_ADVANCED AV_PROFILE_VC1_ADVANCED
#endif
#endif
#ifndef FF_PROFILE_HEVC_MAIN
#ifdef AV_PROFILE_HEVC_MAIN
#define FF_PROFILE_HEVC_MAIN AV_PROFILE_HEVC_MAIN
#define FF_PROFILE_HEVC_MAIN_10 AV_PROFILE_HEVC_MAIN_10
#endif
#endif
#ifndef FF_PROFILE_VP9_0
#ifdef AV_PROFILE_VP9_0
#define FF_PROFILE_VP9_0 AV_PROFILE_VP9_0
#define FF_PROFILE_VP9_1 AV_PROFILE_VP9_1
#define FF_PROFILE_VP9_2 AV_PROFILE_VP9_2
#define FF_PROFILE_VP9_3 AV_PROFILE_VP9_3
#endif
#endif

// AV_CODEC_ID_H265 is a macro defined as AV_CODEC_ID_HEVC in ffmpeg but not in libav. so we can use FF_PROFILE_HEVC_MAIN to avoid libavcodec version check. (from ffmpeg 2.1)
#ifndef FF_PROFILE_HEVC_MAIN //libav does not define it
#define AV_CODEC_ID_HEVC ((AVCodecID)0) //QTAV_CODEC_ID(NONE)
#define CODEC_ID_HEVC ((AVCodecID)0) //QTAV_CODEC_ID(NONE)
#define FF_PROFILE_HEVC_MAIN -1
#define FF_PROFILE_HEVC_MAIN_10 -1
#endif
#if !FFMPEG_MODULE_CHECK(LIBAVCODEC, 54, 92, 100) && !LIBAV_MODULE_CHECK(LIBAVCODEC, 55, 34, 1) //ffmpeg1.2 libav10
#define AV_CODEC_ID_VP9 ((AVCodecID)0) //QTAV_CODEC_ID(NONE)
#define CODEC_ID_VP9 ((AVCodecID)0) //QTAV_CODEC_ID(NONE)
#endif
#ifndef FF_PROFILE_VP9_0
#define FF_PROFILE_VP9_0 0
#define FF_PROFILE_VP9_1 1
#define FF_PROFILE_VP9_2 2
#define FF_PROFILE_VP9_3 3
#endif

#define AV_RUN_CHECK(FUNC, RETURN, ...) do { \
    int ret = FUNC; \
    if (ret < 0) { \
        char str[AV_ERROR_MAX_STRING_SIZE]; \
        memset(str, 0, sizeof(str)); \
        av_strerror(ret, str, sizeof(str)); \
        av_log(NULL, AV_LOG_WARNING, "Error " #FUNC " @%d " __FILE__ ": (%#x) %s\n", __LINE__, ret, str); \
        RETURN __VA_ARGS__; \
     } } while(0)

/*
 * FFmpeg 7.0 (lavu 59) removed the deprecated uint64 channel layout api. Channel layout is a 64bit mask in QtAV,
 * so emulate the removed functions and add accessors for the fields that were replaced by AVChannelLayout ch_layout.
 */
#if LIBAVUTIL_VERSION_MAJOR >= 59
#include <libavutil/channel_layout.h>
static inline int av_get_channel_layout_nb_channels(uint64_t mask)
{
    int n = 0;
    for (; mask; mask &= mask - 1)
        ++n;
    return n;
}

static inline int64_t av_get_default_channel_layout(int nb_channels)
{
    AVChannelLayout cl;
    av_channel_layout_default(&cl, nb_channels);
    const int64_t mask = cl.order == AV_CHANNEL_ORDER_NATIVE ? (int64_t)cl.u.mask : 0;
    av_channel_layout_uninit(&cl);
    return mask;
}

/// nb_channels <= 0: use the number of channels in layout
static inline void av_get_channel_layout_string(char *buf, int buf_size, int nb_channels, uint64_t mask)
{
    AVChannelLayout cl;
    if (mask && av_channel_layout_from_mask(&cl, mask) == 0) {
        av_channel_layout_describe(&cl, buf, buf_size);
        av_channel_layout_uninit(&cl);
        return;
    }
    snprintf(buf, buf_size, "%d channels", nb_channels > 0 ? nb_channels : 0);
}
#endif //LIBAVUTIL_VERSION_MAJOR >= 59

#if LIBAVUTIL_VERSION_MAJOR >= 58 //ch_layout is available since lavu 57.24, but the old fields are only removed in 59
static inline int64_t qtav_channel_layout_mask(const AVChannelLayout &cl, int nb_channels_hint = 0)
{
    if (cl.order == AV_CHANNEL_ORDER_NATIVE)
        return (int64_t)cl.u.mask;
    const int n = cl.nb_channels > 0 ? cl.nb_channels : nb_channels_hint;
    return n > 0 ? av_get_default_channel_layout(n) : 0;
}
static inline void qtav_set_channel_layout(AVChannelLayout *cl, int64_t mask, int nb_channels)
{
    av_channel_layout_uninit(cl);
    if (mask && av_channel_layout_from_mask(cl, (uint64_t)mask) == 0)
        return;
    if (nb_channels > 0)
        av_channel_layout_default(cl, nb_channels);
}
#define QTAV_CH_LAYOUT_API 1
#define QTAV_AVCTX_CHANNELS(ctx) ((ctx)->ch_layout.nb_channels)
#define QTAV_AVCTX_CHANNEL_LAYOUT(ctx) qtav_channel_layout_mask((ctx)->ch_layout)
#define QTAV_AVCTX_SET_CHANNEL_LAYOUT(ctx, mask, channels) qtav_set_channel_layout(&(ctx)->ch_layout, mask, channels)
#define QTAV_AVFRAME_CHANNELS(f) ((f)->ch_layout.nb_channels)
#define QTAV_AVFRAME_CHANNEL_LAYOUT(f) qtav_channel_layout_mask((f)->ch_layout)
#define QTAV_AVFRAME_SET_CHANNEL_LAYOUT(f, mask, channels) qtav_set_channel_layout(&(f)->ch_layout, mask, channels)
#else
#define QTAV_AVCTX_CHANNELS(ctx) ((ctx)->channels)
#define QTAV_AVCTX_CHANNEL_LAYOUT(ctx) ((ctx)->channel_layout)
#define QTAV_AVCTX_SET_CHANNEL_LAYOUT(ctx, mask, nb) do { (ctx)->channel_layout = (mask); (ctx)->channels = (nb); } while (0)
#define QTAV_AVFRAME_CHANNELS(f) ((f)->channels)
#define QTAV_AVFRAME_CHANNEL_LAYOUT(f) ((f)->channel_layout)
#define QTAV_AVFRAME_SET_CHANNEL_LAYOUT(f, mask, nb) do { (f)->channel_layout = (mask); (f)->channels = (nb); } while (0)
#endif

// swr_alloc_set_opts was removed in FFmpeg 7.0 (swr 5)
#if QTAV_HAVE(SWRESAMPLE) && LIBSWRESAMPLE_VERSION_MAJOR >= 5
static inline SwrContext *qtav_swr_alloc_set_opts(SwrContext *s, int64_t out_ch_layout, AVSampleFormat out_sample_fmt, int out_sample_rate
                                                  , int64_t in_ch_layout, AVSampleFormat in_sample_fmt, int in_sample_rate, int log_offset, void *log_ctx)
{
    AVChannelLayout out, in;
    if (av_channel_layout_from_mask(&out, (uint64_t)out_ch_layout) < 0)
        return NULL;
    if (av_channel_layout_from_mask(&in, (uint64_t)in_ch_layout) < 0) {
        av_channel_layout_uninit(&out);
        return NULL;
    }
    SwrContext *ctx = s;
    if (swr_alloc_set_opts2(&ctx, &out, out_sample_fmt, out_sample_rate, &in, in_sample_fmt, in_sample_rate, log_offset, log_ctx) < 0)
        ctx = NULL;
    av_channel_layout_uninit(&out);
    av_channel_layout_uninit(&in);
    return ctx;
}
#define swr_alloc_set_opts qtav_swr_alloc_set_opts
#endif

#endif //QTAV_COMPAT_H

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(56,33,0)
#define AV_CODEC_FLAG_GLOBAL_HEADER CODEC_FLAG_GLOBAL_HEADER
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(56,56,100)
#define AV_INPUT_BUFFER_MIN_SIZE FF_MIN_BUFFER_SIZE
#elif !defined(AV_INPUT_BUFFER_MIN_SIZE) //removed in FFmpeg 7.0
#define AV_INPUT_BUFFER_MIN_SIZE 16384
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(56,56,100)
#define AV_INPUT_BUFFER_PADDING_SIZE FF_INPUT_BUFFER_PADDING_SIZE
#endif
