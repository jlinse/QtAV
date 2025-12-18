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

#ifndef QTAV_SUBTITLEPROCESSOR_H
#define QTAV_SUBTITLEPROCESSOR_H

#include <QtAV/QtAV_Global.h>
#include <QtAV/Subtitle.h>
#include <QtCore/QList>
#include <QtGui/QImage>

namespace QtAV {

class SubtitleProcessorPrivate;
class Q_AV_PRIVATE_EXPORT SubtitleProcessor
{
public:
    SubtitleProcessor();
    virtual ~SubtitleProcessor();
    /*!
     * \brief supportedTypes
     * \return a list of supported suffixes. e.g. [ "ass", "ssa", "srt" ]
     * used to find subtitle files with given suffixes
     */
    QStringList supportedTypes() const;
    /*!
     * \brief process
     * process subtitle from QIODevice.
     * \param dev dev is open and you don't have to close it
     * \return false if failed or does not supports iodevice, e.g. does not support sequential device
     */
    bool process(QIODevice *dev);
    /*!
     * \brief process
     * default behavior is calling process(QFile*)
     * \param path
     * \return false if failed or does not support file
     */
    bool process(const QString &path);
    /*!
     * \brief timestamps
     * call this after process(). SubtitleFrame.text must be set
     * \return
     */
    QList<SubtitleFrame> frames();
    bool canRender() const { return true; }
    // return false if not supported
    bool processHeader(const QByteArray &codec, const QByteArray &data);
    // return timestamp, insert it to Subtitle's internal linkedlist. can be invalid if only support renderering
    SubtitleFrame processLine(const QByteArray &data, qreal pts = -1, qreal duration = 0);
    QString getText(qreal pts) const;
    // default null image
    QImage getImage(qreal pts, QRect *boundingRect = 0);
    SubImageSet getSubImages(qreal pts, QRect *boundingRect = 0);
    void setFrameSize(int width, int height);
    QSize frameSize() const;
    int frameWidth() const;
    int frameHeight() const;

    void setFontFile(const QString &file);
    void setFontsDir(const QString &dir);
    void setFontFileForced(bool force);

private:
    SubtitleProcessorPrivate* d_ptr;
};

} //namespace QtAV
#endif // QTAV_SUBTITLEPROCESSOR_H
