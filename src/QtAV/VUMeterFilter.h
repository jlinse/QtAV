#ifndef OMPLAYER_VUMETERFILTER_HPP
#define OMPLAYER_VUMETERFILTER_HPP

#include <QtAV/Filter.h>

namespace QtAV {

class Q_AV_EXPORT VUMeterFilter : public AudioFilter
{
    Q_OBJECT

public:
    explicit VUMeterFilter(QObject *parent = nullptr);

    inline float leftLevel() const { return left; }
    inline float rightLevel() const { return right; }

Q_SIGNALS:
    void leftLevelChanged(float value);  //dB
    void rightLevelChanged(float value); //dB

protected:
    void process(Statistics *statistics, AudioFrame *frame) override;

private:
    float left = 0, right = 0;
};
} // namespace QtAV

#endif //OMPLAYER_VUMETERFILTER_HPP
