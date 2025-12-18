/******************************************************************************
    QtAV:  Multimedia framework based on Qt and FFmpeg
    Copyright (C) 2012-2022 Wang Bin <wbsecg1@gmail.com>

*   This file is part of QtAV (from 2014)

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

#include "AVWrapper.h"
#include "PlainText.h"
#include "QtAV/AVDemuxer.h"
#include "QtAV/Packet.h"
#include "QtAV/private/AVCompat.h"
#include "utils/Logger.h"
#include "utils/internal.h"
#include <QIODevice>
#include <QtCore/QCoreApplication>
#include <QtCore/QDir>
#include <QtCore/QEventLoop>
#include <QtCore/QFile>
#include <QtCore/QMutex>
#include <QtCore/QThread>

//#define ASS_CAPI_NS // do not unload() manually!
//#define CAPI_LINK_ASS
#include "capi/ass_api.h"
#include <stdarg.h>
//#include <string>  //include after ass_api.h, stdio.h is included there in a different namespace

namespace QtAV {

void RenderASS(QImage *image, const SubImage &img, int dstX, int dstY);

class SubtitleProcessorPrivate : protected ass::api
{
public:
    SubtitleProcessorPrivate();
    ~SubtitleProcessorPrivate();

    bool process(QIODevice *dev);
    // supportsFromFile must be true
    bool process(const QString &path);
    QList<SubtitleFrame> frames() const;
    bool processHeader(const QByteArray &codec, const QByteArray &data);
    SubtitleFrame processLine(const QByteArray &data, qreal pts = -1, qreal duration = 0);
    QString getText(qreal pts) const;
    QImage getImage(qreal pts, QRect *boundingRect = 0);
    SubImageSet getSubImages(qreal pts, QRect *boundingRect = 0);
    void setFontFile(const QString &file);
    void setFontsDir(const QString &dir);
    void setFontFileForced(bool force);
    void setFrameSize(int width, int height);

    bool processSubtitle(AVDemuxer &demuxer);
    void processTrack(ASS_Track *track);

    bool initRenderer();
    void updateFontCache();
    void updateFontCacheAsync();

    bool libassProcessHeader(const QByteArray &codec, const QByteArray &data);
    SubtitleFrame libassProcessLine(const QByteArray &data, qreal pts = -1, qreal duration = 0);
    SubImageSet getSubImages(qreal pts, QRect *boundingRect, QImage *qimg, bool copy);

    int m_width, m_height;

    Wrapper::AVCodecContextWrapper codec_ctx;
    QList<SubtitleFrame> m_frames;

    bool m_update_cache;
    bool force_font_file; // works only iff font_file is set
    QString font_file;
    QString fonts_dir;
    QByteArray m_codec;
    ASS_Library *m_ass;
    ASS_Renderer *m_renderer;
    ASS_Track *m_track;

    //cache the image for the last invocation. return this if image does not change
    QImage m_image;
    SubImageSet m_assimages;
    QRect m_bound;
    mutable QMutex m_mutex;
};

// log level from ass_utils.h
#define MSGL_FATAL 0
#define MSGL_ERR 1
#define MSGL_WARN 2
#define MSGL_INFO 4
#define MSGL_V 6
#define MSGL_DBG2 7

static void ass_msg_cb(int level, const char *fmt, va_list va, void *data)
{
    Q_UNUSED(data)
    if (level > MSGL_INFO)
        return;
#ifdef Q_OS_WIN
    if (level == MSGL_WARN) {
        return; //crash at warnings from fontselect
    }
#endif
    printf("[libass]: ");
    vprintf(fmt, va);
    printf("\n");
    fflush(0);
    return;
    QString msg(QStringLiteral("{libass} ")
                + QString().vsprintf(fmt, va)); //QString.vsprintf() may crash at strlen().
    if (level == MSGL_FATAL)
        qFatal("%s", msg.toUtf8().constData());
    else if (level <= 2)
        qWarning() << msg;
    else if (level <= MSGL_INFO)
        qDebug() << msg;
}

SubtitleProcessorPrivate::SubtitleProcessorPrivate()
    : m_update_cache(true)
    , force_font_file(true)
    , m_ass(0)
    , m_renderer(0)
    , m_track(0)
    , m_width(0)
    , m_height(0)
{
    if (!ass::api::loaded())
        return;
    m_ass = ass_library_init();
    if (!m_ass) {
        qWarning("ass_library_init failed!");
        return;
    }
    ass_set_message_cb(m_ass, ass_msg_cb, NULL);
}
SubtitleProcessorPrivate::~SubtitleProcessorPrivate()
{ // ass dll is loaded if ass objects are available
    if (m_track) {
        ass_free_track(m_track);
        m_track = 0;
    }
    if (m_renderer) {
        QMutexLocker lock(&m_mutex);
        Q_UNUSED(lock);
        ass_renderer_done(m_renderer); // check async update cache!!
        m_renderer = 0;
    }
    if (m_ass) {
        ass_library_done(m_ass);
        m_ass = 0;
    }
}

bool SubtitleProcessorPrivate::process(QIODevice *dev)
{
    if (!dev->isOpen()) {
        if (!dev->open(QIODevice::ReadOnly)) {
            qWarning() << "open qiodevice error: " << dev->errorString();
            return false;
        }
    }
    AVDemuxer demuxer;
    demuxer.setMedia(dev);
    if (!demuxer.load())
        goto error;
    if (demuxer.subtitleStreams().isEmpty())
        goto error;
    qDebug("subtitle format: %s", demuxer.formatContext()->iformat->name);
    if (!processSubtitle(demuxer))
        goto error;
    demuxer.unload();
    return true;
error:
    demuxer.unload();
    return false;
}

bool SubtitleProcessorPrivate::process(const QString &path)
{
    AVDemuxer demuxer;
    demuxer.setMedia(path);
    if (!demuxer.load())
        goto error;
    if (demuxer.subtitleStreams().isEmpty())
        goto error;
    qDebug("subtitle format: %s", demuxer.formatContext()->iformat->name);
    if (!processSubtitle(demuxer))
        goto error;
    demuxer.unload();
    return true;
error:
    demuxer.unload();
    return false;
}

QList<SubtitleFrame> SubtitleProcessorPrivate::frames() const
{
    return m_frames;
}

QString SubtitleProcessorPrivate::getText(qreal pts) const
{
    QMutexLocker lock(&m_mutex);
    Q_UNUSED(lock);
    QString text;
    for (int i = 0; i < m_frames.size(); ++i) {
        if (m_frames[i].begin <= pts && m_frames[i].end >= pts) {
            text += m_frames[i].text + QStringLiteral("\n");
            continue;
        }
        if (!text.isEmpty())
            break;
    }
    return text.trimmed();
}
QImage SubtitleProcessorPrivate::getImage(qreal pts, QRect *boundingRect)
{
    // ass dll is loaded if ass library is available
    getSubImages(pts, boundingRect, &m_image, false);
    m_assimages.reset();
    return m_image;
}
SubImageSet SubtitleProcessorPrivate::getSubImages(qreal pts, QRect *boundingRect)
{
    m_assimages = getSubImages(pts, boundingRect, NULL, true);
    return m_assimages;
}
SubImageSet SubtitleProcessorPrivate::getSubImages(qreal pts,
                                                   QRect *boundingRect,
                                                   QImage *qimg,
                                                   bool copy)
{ // ass dll is loaded if ass library is available
    {
        QMutexLocker lock(&m_mutex);
        Q_UNUSED(lock);
        if (!m_ass) {
            qWarning("ass library not available");
            return SubImageSet();
        }
        if (!m_track) {
            qWarning("ass track not available");
            return SubImageSet();
        }
        if (!m_renderer) {
            initRenderer();
            if (!m_renderer) {
                qWarning("ass renderer not available");
                return SubImageSet();
            }
        }
    }
    if (m_update_cache)
        updateFontCache();

    QMutexLocker lock(&m_mutex);
    Q_UNUSED(lock);
    if (!m_renderer) //reset in setFontXXX
        return SubImageSet();
    int detect_change = 0;
    ASS_Image *img = ass_render_frame(m_renderer,
                                      m_track,
                                      (long long) (pts * 1000.0),
                                      &detect_change);
    if (!detect_change && !m_assimages.isValid()) {
        if (boundingRect)
            *boundingRect = m_bound;
        return m_assimages;
    }
    m_image = QImage();
    m_assimages.reset(m_width, m_height, SubImageSet::ASS);
    QRect rect(0, 0, 0, 0);
    ASS_Image *i = img;
    while (i) {
        const quint8 a = 255 - (i->color & 0xff);
        //qDebug("ass %d rect %d: %d,%d-%dx%d", a, n, i->dst_x, i->dst_y, i->w, i->h);
        if (i->w <= 0 || i->h <= 0 || a == 0) {
            i = i->next;
            continue;
        }
        SubImage s(i->dst_x, i->dst_y, i->w, i->h, i->stride);
        s.color = i->color;
        if (copy) {
            s.data.reserve(i->stride * i->h);
            s.data.resize(i->stride * i->h);
            memcpy(s.data.data(), i->bitmap, i->stride * (i->h - 1) + i->w);
        } else {
            s.data = QByteArray::fromRawData((const char *) i->bitmap,
                                             i->stride * (i->h - 1) + i->w);
        }
        m_assimages.images.append(s);
        rect |= QRect(i->dst_x, i->dst_y, i->w, i->h);
        i = i->next;
    }
    m_bound = rect;
    if (boundingRect) {
        *boundingRect = m_bound;
    }
    if (!qimg)
        return m_assimages;
    *qimg = QImage(rect.size(), QImage::Format_ARGB32);
    qimg->fill(Qt::transparent);
    foreach (const SubImage &i, m_assimages.images) {
        RenderASS(qimg, i, i.x - rect.x(), i.y - rect.y());
    }
    return m_assimages;
}

void SubtitleProcessorPrivate::setFontFile(const QString &file)
{
    if (font_file == file)
        return;
    font_file = file;
    m_update_cache = true; //update renderer when getting the next image
    if (m_renderer) {
        QMutexLocker lock(&m_mutex);
        Q_UNUSED(lock);
        // resize frame to ensure renderer can be resized later
        setFrameSize(-1, -1);
        ass_renderer_done(m_renderer);
        m_renderer = 0;
    }
}

void SubtitleProcessorPrivate::setFontsDir(const QString &dir)
{
    if (fonts_dir == dir)
        return;
    fonts_dir = dir;
    m_update_cache = true; //update renderer when getting the next image
    if (m_renderer) {
        QMutexLocker lock(&m_mutex);
        Q_UNUSED(lock);
        // resize frame to ensure renderer can be resized later
        setFrameSize(-1, -1);
        ass_renderer_done(m_renderer);
        m_renderer = 0;
    }
}

void SubtitleProcessorPrivate::setFontFileForced(bool force)
{
    if (force_font_file == force)
        return;
    force_font_file = force;
    // FIXME: sometimes crash
    m_update_cache = true; //update renderer when getting the next image
    if (m_renderer) {
        QMutexLocker lock(&m_mutex);
        Q_UNUSED(lock);
        // resize frame to ensure renderer can be resized later
        setFrameSize(-1, -1);
        ass_renderer_done(m_renderer);
        m_renderer = 0;
    }
}

void SubtitleProcessorPrivate::setFrameSize(int width, int height)
{
    if (width == m_width && height == m_height)
        return;
    m_width = width;
    m_height = height;

    if (width < 0 || height < 0)
        return;
    if (!m_renderer) {
        initRenderer();
    }
    if (!m_renderer)
        return;
    ass_set_frame_size(m_renderer, width, height);
}

bool SubtitleProcessorPrivate::processHeader(const QByteArray &codec, const QByteArray &data)
{
    Q_UNUSED(data);

    const AVCodec *c = avcodec_find_decoder_by_name(codec.constData());
    if (!c) {
        qDebug("subtitle avcodec_descriptor_get_by_name %s", codec.constData());
        const AVCodecDescriptor *desc = avcodec_descriptor_get_by_name(codec.constData());
        if (!desc) {
            qWarning("No codec descriptor found for %s", codec.constData());
            return false;
        }
        c = avcodec_find_decoder(desc->id);
    }
    if (!c) {
        qWarning("No subtitle decoder found for codec: %s, try fron descriptor", codec.constData());
        return false;
    }
    codec_ctx.assign(c);
    if (!codec_ctx)
        return false;

    if (!data.isEmpty()) {
        av_free(codec_ctx->extradata);
        codec_ctx->extradata = (uint8_t *) av_mallocz(data.size() + AV_INPUT_BUFFER_PADDING_SIZE);
        if (!codec_ctx->extradata)
            return false;
        codec_ctx->extradata_size = data.size();
        memcpy(codec_ctx->extradata, data.constData(), data.size());
    }
    if (avcodec_open2(codec_ctx, c, NULL) < 0) {
        codec_ctx.reset();
        return false;
    }
    return libassProcessHeader("ass", (const char *) codec_ctx->subtitle_header);
    //return true;//codec != QByteArrayLiteral("ass") && codec != QByteArrayLiteral("ssa");
}

SubtitleFrame SubtitleProcessorPrivate::processLine(const QByteArray &data,
                                                    qreal pts,
                                                    qreal duration)
{
    //qDebug() << "line: " << data;
    if (!codec_ctx) {
        return SubtitleFrame();
    }
    Wrapper::AVPacketWrapper packet;

    packet->size = data.size();
    packet->data = (uint8_t *) data.constData();

    AVSubtitle sub;
    memset(&sub, 0, sizeof(sub));
    int got_subtitle = 0;
    int ret = avcodec_decode_subtitle2(codec_ctx, &sub, &got_subtitle, &packet);
    if (ret < 0 || !got_subtitle) {
        avsubtitle_free(&sub);
        return SubtitleFrame();
    }

    SubtitleFrame frame;
    // start_display_time and duration are in ms
    frame.begin = pts;
    frame.end = pts + duration;
    //qDebug() << QTime(0, 0, 0).addMSecs(frame.begin*1000.0) << "-" << QTime(0, 0, 0).addMSecs(frame.end*1000.0) << " fmt: " << sub.format << " pts: " << m_reader.packet().pts //sub.pts
    //       << " rects: " << sub.num_rects << " end: " << sub.end_display_time;
    for (unsigned i = 0; i < sub.num_rects; i++) {
        switch (sub.rects[i]->type) {
        case SUBTITLE_ASS:
            frame = libassProcessLine(sub.rects[i]->ass, pts, duration);
            break;
        case SUBTITLE_TEXT:
            //qDebug("txt frame: %s", sub.rects[i]->text);
            frame.text.append(QString::fromUtf8(sub.rects[i]->text)).append(ushort('\n'));
            break;
        case SUBTITLE_BITMAP:
            //sub.rects[i]->w > 0 && sub.rects[i]->h > 0
            //qDebug("bmp sub");
            frame = SubtitleFrame(); // not support bmp subtitle now
            break;
        default:
            break;
        }
    }
    avsubtitle_free(&sub);
    return frame;
}
bool SubtitleProcessorPrivate::libassProcessHeader(const QByteArray &codec, const QByteArray &data)
{
    if (!ass::api::loaded())
        return false;
    QMutexLocker lock(&m_mutex);
    Q_UNUSED(lock);
    m_codec = codec;
    m_frames.clear();
    setFrameSize(-1, -1);
    if (m_track) {
        ass_free_track(m_track);
        m_track = 0;
    }
    m_track = ass_new_track(m_ass);
    if (!m_track) {
        qWarning("failed to create an ass track");
        return false;
    }
    ass_process_codec_private(m_track, (char *) data.constData(), data.size());
    return true;
}
SubtitleFrame SubtitleProcessorPrivate::libassProcessLine(const QByteArray &data,
                                                          qreal pts,
                                                          qreal duration)
{
    if (!ass::api::loaded())
        return SubtitleFrame();
    if (data.isEmpty() || data.at(0) == 0)
        return SubtitleFrame();
    QMutexLocker lock(&m_mutex);
    Q_UNUSED(lock);
    if (!m_track)
        return SubtitleFrame();
    const int nb_tracks = m_track->n_events;
    // TODO: confirm. ass/ssa path from mpv
    if (m_codec == QByteArrayLiteral("ass")) {
        ass_process_chunk(m_track,
                          (char *) data.constData(),
                          data.size(),
                          pts * 1000.0,
                          duration * 1000.0);
    } else { //ssa
        //ssa. mpv: flush_on_seek, broken ffmpeg ASS packet format
        ass_process_data(m_track, (char *) data.constData(), data.size());
    }
    if (nb_tracks == m_track->n_events)
        return SubtitleFrame();
    //qDebug("events: %d", m_track->n_events);
    for (int i = m_track->n_events - 1; i >= 0; --i) {
        const ASS_Event &ae = m_track->events[i];
        //qDebug("ass_event[%d] %lld+%lld/%lld+%lld: %s", i, ae.Start, ae.Duration, (long long)(pts*1000.0),  (long long)(duration*1000.0), ae.Text);
        //packet.duration can be 0
        if (ae.Text == nullptr)
            continue;
        if (ae.Start
            == (long long) (pts * 1000.0)) { // && ae.Duration == (long long)(duration*1000.0)) {
            SubtitleFrame frame;
            frame.text = PlainText::fromAss(ae.Text);
            frame.begin = qreal(ae.Start) / 1000.0;
            frame.end = frame.begin + qreal(ae.Duration) / 1000.0;
            return frame;
        }
    }
    return SubtitleFrame();
}
bool SubtitleProcessorPrivate::processSubtitle(AVDemuxer &demuxer)
{
    m_frames.clear();
    int ss = demuxer.subtitleStream();
    if (ss < 0) {
        qWarning("no subtitle stream found");
        return false;
    }
    codec_ctx = demuxer.playSubtitleCodecContext();

    const AVCodec *dec = avcodec_find_decoder(codec_ctx->codec_id);
    const AVCodecDescriptor *dec_desc = avcodec_descriptor_get(codec_ctx->codec_id);
    if (!dec) {
        if (dec_desc)
            qWarning("Failed to find subtitle codec %s", dec_desc->name);
        else
            qWarning("Failed to find subtitle codec %d", codec_ctx->codec_id);
        return false;
    }
    qDebug("found subtitle decoder '%s'", dec_desc->name);
    // AV_CODEC_PROP_TEXT_SUB: ffmpeg >= 2.0
#ifdef AV_CODEC_PROP_TEXT_SUB
    if (dec_desc && !(dec_desc->props & AV_CODEC_PROP_TEXT_SUB)) {
        qWarning("Only text based subtitles are currently supported");
        return false;
    }
#endif
    AVDictionary *codec_opts = NULL;
    if (LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 26, 100))
        av_dict_set(&codec_opts, "sub_text_format", "ass", 0);
    int ret = avcodec_open2(codec_ctx, dec, &codec_opts);
    if (ret < 0) {
        qWarning("open subtitle codec error: %s", av_err2str(ret));
        av_dict_free(&codec_opts);
        return false;
    }
    libassProcessHeader("ass", (const char *) codec_ctx->subtitle_header);

    while (!demuxer.atEnd()) {
        if (!demuxer.readFrame()) { // eof or other errors
            continue;
        }
        if (demuxer.stream() != ss)
            continue;
        const Packet pkt = demuxer.packet();
        if (!pkt.isValid())
            continue;
        auto frame = processLine(pkt.data, pkt.pts, pkt.duration);
        if (frame.isValid())
            m_frames.append(frame);
    }
    avcodec_close(codec_ctx);
    codec_ctx.reset();
    return true;
}

void SubtitleProcessorPrivate::processTrack(ASS_Track *track)
{
    // language, track type
    m_frames.clear();
    for (int i = 0; i < track->n_events; ++i) {
        SubtitleFrame frame;
        const ASS_Event &ae = track->events[i];
        frame.text = PlainText::fromAss(ae.Text);
        frame.begin = qreal(ae.Start) / 1000.0;
        frame.end = frame.begin + qreal(ae.Duration) / 1000.0;
        m_frames.append(frame);
    }
}

bool SubtitleProcessorPrivate::initRenderer()
{ //ass_set_extract_fonts(m_ass, 1);
    //ass_set_style_overrides(m_ass, 0);
    m_renderer = ass_renderer_init(m_ass);
    if (!m_renderer) {
        qWarning("ass_renderer_init failed!");
        return false;
    }
#if LIBASS_VERSION >= 0x01000000
    ass_set_shaper(m_renderer, ASS_SHAPING_SIMPLE);
#endif
    return true;
}
// TODO: set font cache dir. default is working dir which may be not writable on some platforms
void SubtitleProcessorPrivate::updateFontCache()
{ // ass dll is loaded if renderer is valid
    QMutexLocker lock(&m_mutex);
    Q_UNUSED(lock);
    if (!m_renderer)
        return;
    // fonts in assets and qrc may change. so check before appFontsDir
    static const QStringList kFontsDirs = QStringList() << qApp->applicationDirPath().append(
                                              QLatin1String("/fonts"))
                                                        << qApp->applicationDirPath() // for winrt
                                                        << QStringLiteral("assets:/fonts")
                                                        << QStringLiteral(":/fonts")
                                                        << Internal::Path::appFontsDir()
#if !defined(Q_OS_WIN32) && !defined(Q_OS_WINRT)
                                                        << Internal::Path::fontsDir()
#endif
        ;
    // TODO: modify fontconfig cache dir in fonts.conf <dir></dir> then save to conf
    static QString conf(0, QChar()); //FC_CONFIG_FILE?
    if (conf.isEmpty() && !conf.isNull()) {
        static const QString kFontCfg(QStringLiteral("fonts.conf"));
        foreach (const QString &fdir, kFontsDirs) {
            qDebug() << "looking up " << kFontCfg << " in: " << fdir;
            QFile cfg(QStringLiteral("%1/%2").arg(fdir).arg(kFontCfg));
            if (!cfg.exists())
                continue;
            conf = cfg.fileName();
            if (fdir.isEmpty() || fdir.startsWith(QLatin1String("assets:"), Qt::CaseInsensitive)
                || fdir.startsWith(QLatin1String(":"), Qt::CaseInsensitive)
                || fdir.startsWith(QLatin1String("qrc:"), Qt::CaseInsensitive)) {
                conf = QStringLiteral("%1/%2").arg(Internal::Path::appFontsDir()).arg(kFontCfg);
                qDebug() << "Fonts dir (for config) is not supported by libass. Copy fonts to app "
                            "fonts dir: "
                         << fdir;
                if (!QDir(Internal::Path::appFontsDir()).exists()) {
                    if (!QDir().mkpath(Internal::Path::appFontsDir())) {
                        qWarning("Failed to create fonts dir: %s",
                                 Internal::Path::appFontsDir().toUtf8().constData());
                    }
                }
                QFile cfgout(conf);
                if (cfgout.exists() && cfgout.size() != cfg.size()) { // TODO:
                    qDebug() << "new " << kFontCfg
                             << " with the same name. remove old: " << cfgout.fileName();
                    cfgout.remove();
                }
                if (!cfgout.exists() && !cfg.copy(conf)) {
                    qWarning() << "Copy font config file [" << cfg.fileName()
                               << "] error: " << cfg.errorString();
                    continue;
                }
            }
            break;
        }
        if (!QFile(conf).exists())
            conf.clear();
        qDebug() << "FontConfig: " << conf;
    }
    /*
     * Fonts dir look up:
     * - appdir/fonts has fonts
     * - assets:/fonts, qrc:/fonts: copy fonts to appFontsDir
     * - appFontsDir (appdir/fonts has no fonts)
     * - fontsDir if it has font files
     * If defaults.ttf exists in fonts dir(for 'assets:' and 'qrc:' appFontsDir is checked), use it as default font and disable font provider
     * (for example fontconfig) to speed up(skip) libass font look up.
     * Skip setting fonts dir
     */
    static QString sFont(0,
                         QChar()); // if exists, fontconfig will be disabled and directly use this font
    static QString sFontsDir(0, QChar());
    if (sFontsDir.isEmpty() && !sFontsDir.isNull()) {
        static const QString kDefaultFontName(QStringLiteral("default.ttf"));
        static const QStringList ft_filters = QStringList()
                                              << QStringLiteral("*.ttf") << QStringLiteral("*.otf")
                                              << QStringLiteral("*.ttc");
        QStringList fonts;
        foreach (const QString &fdir, kFontsDirs) {
            qDebug() << "looking up fonts in: " << fdir;
            QDir d(fdir);
            if (!d.exists()) //avoid winrt crash (system fonts dir)
                continue;
            fonts = d.entryList(ft_filters, QDir::Files);
            if (fonts.isEmpty())
                continue;
            if (fonts.contains(kDefaultFontName)) {
                QFile ff(QStringLiteral("%1/%2").arg(fdir).arg(kDefaultFontName));
                if (ff.exists() && ff.size() == 0) {
                    qDebug("invalid default font");
                    fonts.clear();
                    continue;
                }
            }
            sFontsDir = fdir;
            break;
        }
        if (fonts.isEmpty()) {
            sFontsDir.clear();
        } else {
            qDebug() << "fonts dir: " << sFontsDir << "  font files: " << fonts;
            if (sFontsDir.isEmpty()
                || sFontsDir.startsWith(QLatin1String("assets:"), Qt::CaseInsensitive)
                || sFontsDir.startsWith(QLatin1String(":"), Qt::CaseInsensitive)
                || sFontsDir.startsWith(QLatin1String("qrc:"), Qt::CaseInsensitive)) {
                const QString fontsdir_in(sFontsDir);
                sFontsDir = Internal::Path::appFontsDir();
                qDebug() << "Fonts dir is not supported by libass. Copy fonts to app fonts dir if "
                            "not exist: "
                         << sFontsDir;
                if (!QDir(Internal::Path::appFontsDir()).exists()) {
                    if (!QDir().mkpath(Internal::Path::appFontsDir())) {
                        qWarning("Failed to create fonts dir: %s",
                                 Internal::Path::appFontsDir().toUtf8().constData());
                    }
                }
                foreach (const QString &f, fonts) {
                    QFile ff(QStringLiteral("%1/%2").arg(fontsdir_in).arg(f));
                    const QString kOut(QStringLiteral("%1/%2").arg(sFontsDir).arg(f));
                    QFile ffout(kOut);
                    if (ffout.exists() && ffout.size() != ff.size()) { // TODO:
                        qDebug() << "new font with the same name. remove old: " << ffout.fileName();
                        ffout.remove();
                    }
                    if (!ffout.exists() && !ff.copy(kOut))
                        qWarning() << "Copy font file [" << ff.fileName()
                                   << "] error: " << ff.errorString();
                }
            }
            if (fonts.contains(kDefaultFontName)) {
                sFont = QStringLiteral("%1/%2").arg(sFontsDir).arg(kDefaultFontName);
                qDebug() << "default font file: " << sFont << "; fonts dir: " << sFontsDir;
            }
        }
    }
    static QByteArray family; //fallback to Arial?
    if (family.isEmpty()) {
        family = qgetenv("QTAV_SUB_FONT_FAMILY_DEFAULT");
        //Setting default font to the Arial from default.ttf (used if FontConfig fails)
        if (family.isEmpty())
            family = QByteArrayLiteral("Arial");
    }
    // prefer user settings
    const QString kFont = font_file.isEmpty() ? sFont : Internal::Path::toLocal(font_file);
    const QString kFontsDir = fonts_dir.isEmpty() ? sFontsDir : Internal::Path::toLocal(fonts_dir);
    qDebug() << "font file: " << kFont << "; fonts dir: " << kFontsDir;
    // setup libass
#ifdef Q_OS_WINRT
    if (!kFontsDir.isEmpty())
        qDebug("BUG: winrt libass set a valid fonts dir results in crash. skip fonts dir setup.");
#else
    // will call strdup, so safe to use temp array .toUtf8().constData()
    if (!force_font_file || (!font_file.isEmpty() && !QFile::exists(kFont)))
        ass_set_fonts_dir(
            m_ass,
            kFontsDir.isEmpty()
                ? 0
                : kFontsDir.toUtf8()
                      .constData()); // look up fonts in fonts dir can be slow. force font file to skip lookup
#endif
    /* ass_set_fonts:
     * fc/dfp=false(auto font provider): Prefer font provider to find a font(FC needs fonts.conf) in font_dir, or provider's configuration. If failed, try the given font
     * fc/dfp=true(no font provider): only try the given font
     */
    // user can prefer font provider(force_font_file=false), or disable font provider to force the given font
    // if provider is enabled, libass can fallback to the given font if provider can not provide a font
    const QByteArray a_conf(conf.toUtf8());
    const char *kConf = conf.isEmpty() ? 0 : a_conf.constData();
    if (kFont.isEmpty()) { // TODO: always use font provider if no font file is set, i.e. ignore force_font_file
        qDebug("No font file is set, use font provider");
        ass_set_fonts(m_renderer, NULL, family.constData(), !force_font_file, kConf, 1);
    } else {
        qDebug("Font file is set. force font file: %d", force_font_file);
        ass_set_fonts(m_renderer,
                      kFont.toUtf8().constData(),
                      family.constData(),
                      !force_font_file,
                      kConf,
                      1);
    }
    m_update_cache = false; //TODO: set true if user set a new font or fonts dir}

} //namespace QtAV

void SubtitleProcessorPrivate::updateFontCacheAsync()
{
    class FontCacheUpdater : public QThread
    {
        SubtitleProcessorPrivate *sp;

    public:
        FontCacheUpdater(SubtitleProcessorPrivate *p)
            : sp(p)
        {}
        void run()
        {
            if (!sp)
                return;
            sp->updateFontCache();
        }
    };
    FontCacheUpdater updater(this);
    QEventLoop loop;
    //QObject::connect(&updater, SIGNAL(finished()), &loop, SLOT(quit()));
    updater.start();
    while (updater.isRunning()) {
        loop.processEvents();
    }
    //loop.exec(); // what if updater is finished before exec()?
    //updater.wait();
}

} // namespace QtAV