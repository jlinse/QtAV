#pragma once
#include <QtGlobal>

struct AVFrame;
struct AVPacket;
struct AVCodecContext;
struct AVCodecParameters;
struct AVCodec;

namespace QtAV::Wrapper {

	class AVPacketWrapper {
	public:
		AVPacketWrapper();
		AVPacketWrapper(const AVPacket* packet);
		~AVPacketWrapper();

	public:
		AVPacketWrapper(const AVPacketWrapper& other);
		AVPacketWrapper& operator=(const AVPacketWrapper& other);

		const ::AVPacket* operator&() const;
		::AVPacket* operator&();

		const ::AVPacket* data() const;
		::AVPacket* data();

		::AVPacket* operator->() const;

		int calculatePacketSize() const;
	private:
		::AVPacket* packet_;
	};

	class AVFrameWapper
	{
	public:
		AVFrameWapper();
		AVFrameWapper(const AVFrame* frame);
		~AVFrameWapper();
	public:

		AVFrameWapper(const AVFrameWapper& other);
		AVFrameWapper(AVFrameWapper&& other);
		AVFrameWapper& operator=(const AVFrameWapper& other);
		AVFrameWapper& operator=(AVFrameWapper&& other);

		const AVFrame* data() const { return frame_; }
		AVFrame* data() { return frame_; }

		const AVFrame* operator&() const { return frame_; }
		AVFrame* operator&() { return frame_; }

		AVFrame* operator->() const { return frame_; }

		qreal getDAR(const AVCodecContext* codec_ctx) const;
		qreal timestamp() const;

		void reset();
	private:
		AVFrame* frame_;
	};

	class AVCodecContextWrapper {
	public:
		AVCodecContextWrapper() = default;
		AVCodecContextWrapper(const AVCodecParameters* par);
		AVCodecContextWrapper(const AVCodec* codec);
		AVCodecContextWrapper(const AVCodecContext* ctx);
		~AVCodecContextWrapper();
	public:
		AVCodecContextWrapper& operator=(const AVCodecParameters* par);
		AVCodecContextWrapper& operator=(const AVCodecContext* ctx);

		void assign(const AVCodecParameters* par);
		void assign(const AVCodecContext* ctx);
		void assign(const AVCodec* codec);

		void reset();

		void to_avcodec_parameters(AVCodecParameters* par);

		const AVCodecContext* data() const;
		AVCodecContext* data();

		const AVCodecContext* operator&() const;
		AVCodecContext* operator&();

		operator bool() const { return ctx_ != nullptr; }
		operator AVCodecContext* () const { return ctx_; }

		AVCodecContext* operator->() const { return ctx_; }
	private:
		AVCodecContext* ctx_ = nullptr;
	};

}