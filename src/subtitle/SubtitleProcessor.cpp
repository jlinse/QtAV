/******************************************************************************
    QtAV:  Multimedia framework based on Qt and FFmpeg
    Copyright (C) 2012-2016 Wang Bin <wbsecg1@gmail.com>

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

#include "QtAV/private/SubtitleProcessor.h"
#include "QtAV/AVDemuxer.h"
#include "QtAV/private/AVCompat.h"
#include "QtAV/private/factory.h"
#include "SubtitleProcessor_p.h"
#include "utils/Logger.h"

#include <QtCore/QFile>

namespace QtAV {
static QStringList ffmpeg_supported_sub_extensions_by_codec()
{
    QStringList exts;
    const AVCodec *c = NULL;
#if AVCODEC_STATIC_REGISTER
    void *it = NULL;
    while ((c = av_codec_iterate(&it))) {
#else
    avcodec_register_all();
    while ((c = av_codec_next(c))) {
#endif
        if (c->type != AVMEDIA_TYPE_SUBTITLE)
            continue;
        qDebug("sub codec: %s", c->name);
#if AVFORMAT_STATIC_REGISTER
        const AVInputFormat *i = NULL;
        void *it2 = NULL;
        while ((i = av_demuxer_iterate(&it2))) {
#else
        av_register_all(); // MUST register all input/output formats
        AVInputFormat *i = NULL;
        while ((i = av_iformat_next(i))) {
#endif
            if (!strcmp(i->name, c->name)) {
                qDebug("found iformat");
                if (i->extensions) {
                    exts.append(QString::fromLatin1(i->extensions).split(QLatin1Char(',')));
                } else {
                    qDebug("has no exts");
                    exts.append(QString::fromLatin1(i->name));
                }
                break;
            }
        }
        if (!i) {
            // qDebug("codec name '%s' is not found in AVInputFormat, just append codec name", c->name);
            // exts.append(c->name);
        }
    }
    return exts;
}
static QStringList ffmpeg_supported_sub_extensions()
{
    QStringList exts;
#if AVFORMAT_STATIC_REGISTER
    const AVInputFormat *i = NULL;
    void *it = NULL;
    while ((i = av_demuxer_iterate(&it))) {
#else
    av_register_all(); // MUST register all input/output formats
    AVInputFormat *i = NULL;
    while ((i = av_iformat_next(i))) {
#endif
        // strstr parameters can not be null
        if (i->long_name && strstr(i->long_name, "subtitle")) {
            if (i->extensions) {
                exts.append(QString::fromLatin1(i->extensions).split(QLatin1Char(',')));
            } else {
                exts.append(QString::fromLatin1(i->name));
            }
        }
    }
    // AVCodecDescriptor.name and AVCodec.name may be different. avcodec_get_name() use AVCodecDescriptor if possible
    QStringList codecs;
    const AVCodec *c = NULL;
#if AVCODEC_STATIC_REGISTER
    it = NULL;
    while ((c = av_codec_iterate(&it))) {
#else
    avcodec_register_all();
    while ((c = av_codec_next(c))) {
#endif
        if (c->type == AVMEDIA_TYPE_SUBTITLE)
            codecs.append(QString::fromLatin1(c->name));
    }
    const AVCodecDescriptor *desc = NULL;
    while ((desc = avcodec_descriptor_next(desc))) {
        if (desc->type == AVMEDIA_TYPE_SUBTITLE)
            codecs.append(QString::fromLatin1(desc->name));
    }
    exts << codecs;
    exts.removeDuplicates();
    return exts;
}

SubtitleProcessor::SubtitleProcessor()
    : d_ptr(new QtAV::SubtitleProcessorPrivate())
{}
SubtitleProcessor::~SubtitleProcessor()
{
    delete d_ptr;
}

bool SubtitleProcessor::process(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        qWarning() << "open subtitle file error: " << f.errorString();
        return false;
    }
    bool ok = process(&f);
    f.close();
    return ok;
}
QList<SubtitleFrame> SubtitleProcessor::frames()
{
    return d_ptr->frames();
}
QStringList SubtitleProcessor::supportedTypes() const
{
    static const QStringList sSuffixes = ffmpeg_supported_sub_extensions();
    return sSuffixes;
}
bool SubtitleProcessor::process(QIODevice *dev)
{
    return d_ptr->process(dev);
}

QImage SubtitleProcessor::getImage(qreal pts, QRect *boundingRect)
{
    return d_ptr->getImage(pts, boundingRect);
}

SubImageSet SubtitleProcessor::getSubImages(qreal pts, QRect *boundingRect)
{
    return d_ptr->getSubImages(pts, boundingRect);
}

void SubtitleProcessor::setFrameSize(int width, int height)
{
    d_ptr->setFrameSize(width, height);
}

QSize SubtitleProcessor::frameSize() const
{
    return QSize(d_ptr->m_width, d_ptr->m_height);
}

int SubtitleProcessor::frameWidth() const
{
    return d_ptr->m_width;
}

int SubtitleProcessor::frameHeight() const
{
    return d_ptr->m_height;
}

void SubtitleProcessor::setFontFile(const QString &file)
{
    d_ptr->setFontFile(file);
}

void SubtitleProcessor::setFontsDir(const QString &dir)
{
    d_ptr->setFontsDir(dir);
}
void SubtitleProcessor::setFontFileForced(bool force)
{
    d_ptr->setFontFileForced(force);
}

bool SubtitleProcessor::processHeader(const QByteArray &codec, const QByteArray &data)
{
    return d_ptr->processHeader(codec, data);
}

QtAV::SubtitleFrame SubtitleProcessor::processLine(const QByteArray &data,
                                                   qreal pts /*= -1*/,
                                                   qreal duration /*= 0*/)
{
    return d_ptr->processLine(data, pts, duration);
}

QString SubtitleProcessor::getText(qreal pts) const
{
    return d_ptr->getText(pts);
}

} // namespace QtAV
