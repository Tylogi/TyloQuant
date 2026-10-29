// Native checks for the recurrent-mixer operators that Mamba-style linear
// attention layers need: the selective state scan (decode step and any-length
// prefill), the grouped gated RMSNorm and the relu^2 activation.
//
// Every case compares against a double-precision CPU reference implementation
// of the same mathematics, so a wrong grouping, a wrong gate order or a wrong
// dt clamp is caught numerically rather than by inspection.

#include "mfq_hip_activation_ops.h"
#include "mfq_hip_context.h"
#include "mfq_hip_norm_ops.h"
#include "mfq_hip_selective_state_scan_ops.h"
#include "mfq_tensor_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace tb = mfq_tensor_backend;
using tb::Tensor;

constexpr double kUnboundedLimit = std::numeric_limits<double>::infinity();

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void close_value(float actual, double expected, float atol, float rtol,
                 const std::string& label)
{
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > atol + rtol * std::abs(expected)) {
        throw std::runtime_error(label + " mismatch: actual=" +
            std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

void close_vector(const std::vector<float>& actual,
                  const std::vector<double>& expected, float atol, float rtol,
                  const std::string& label)
{
    require(actual.size() == expected.size(), label + " size mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        close_value(actual[index], expected[index], atol, rtol,
            label + "[" + std::to_string(index) + "]");
    }
}

float peak_difference(const std::vector<float>& actual,
                      const std::vector<double>& expected)
{
    require(actual.size() == expected.size(), "difference size mismatch");
    float peak = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        peak = std::max(peak, (float)std::abs(actual[index] - expected[index]));
    }
    return peak;
}

std::vector<double> widen(const std::vector<float>& values)
{
    return std::vector<double>(values.begin(), values.end());
}

std::vector<float> narrow(const std::vector<double>& values)
{
    std::vector<float> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(),
        [](double value) { return (float)value; });
    return result;
}

// Deterministic, well-spread test data inside [-amplitude, amplitude].
std::vector<float> pattern(std::size_t count, float amplitude, float salt)
{
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = amplitude * std::sin(float(index) * 0.173f + salt) +
            0.25f * amplitude * std::cos(float(index) * 0.061f - 3.0f * salt);
    }
    return values;
}

Tensor upload(const std::vector<float>& values,
              std::initializer_list<std::int64_t> shape)
{
    return tb::tensor(values).reshape(shape).to(tb::kCUDA).contiguous();
}

std::vector<float> download(const Tensor& tensor)
{
    auto host = tensor.contiguous().to(tb::kFloat32).cpu();
    return std::vector<float>(
        host.data_ptr<float>(), host.data_ptr<float>() + host.numel());
}

// Round through the storage dtype so the reference sees exactly the values the
// kernel reads: storage quantization belongs to the caller, not to the op.
std::vector<float> round_to(const std::vector<float>& values,
                            std::initializer_list<std::int64_t> shape,
                            tb::ScalarType dtype)
{
    if (dtype == tb::kFloat32) {
        return values;
    }
    return download(upload(values, shape).to(dtype));
}

Tensor upload_as(const std::vector<float>& values,
                 std::initializer_list<std::int64_t> shape,
                 tb::ScalarType dtype)
{
    return upload(values, shape).to(dtype).contiguous();
}

// ---------------------------------------------------------------------------
// Selective state scan
// ---------------------------------------------------------------------------

struct ScanShape {
    int batches = 1;
    int tokens = 1;
    int heads = 1;
    int columns = 1;
    int state_width = 32;
    int groups = 1;
};

struct ScanData {
    ScanShape shape;
    std::vector<double> x;   // [B,T,H,P]
    std::vector<double> dt;  // [B,T,H]
    std::vector<double> b;   // [B,T,G,N]
    std::vector<double> c;   // [B,T,G,N]
    std::vector<float> dt_bias;  // [H] f32 parameter
    std::vector<float> a_log;    // [H] f32 parameter
    std::vector<float> d;        // [H] f32 parameter
};

struct ScanReference {
    std::vector<double> y;      // [B,T,H,P]
    std::vector<double> state;  // [B,H,P,N] after the last token
    std::vector<double> steps;  // [T][B,H,P,N] state after each token
};

double reference_dt(double raw, double bias, double limit_lo, double limit_hi)
{
    const double shifted = raw + bias;
    const double softplus = shifted > 0.0
        ? shifted + std::log1p(std::exp(-shifted))
        : std::log1p(std::exp(shifted));
    return std::min(std::max(softplus, limit_lo), limit_hi);
}

ScanReference reference_scan(const ScanData& data,
                             const std::vector<double>& initial_state,
                             double limit_lo, double limit_hi)
{
    const ScanShape& shape = data.shape;
    const int heads_per_group = shape.heads / shape.groups;
    require(shape.heads % shape.groups == 0, "reference head/group mismatch");
    require(initial_state.size() == (std::size_t)shape.batches * shape.heads *
        shape.columns * shape.state_width, "reference initial state size");
    std::vector<double> state = initial_state;
    const std::size_t plane =
        (std::size_t)shape.heads * shape.columns * shape.state_width;
    ScanReference result;
    result.y.assign((std::size_t)shape.batches * shape.tokens * shape.heads *
        shape.columns, 0.0);
    result.steps.assign((std::size_t)shape.tokens * state.size(), 0.0);
    for (int batch = 0; batch < shape.batches; ++batch) {
        for (int token = 0; token < shape.tokens; ++token) {
            for (int head = 0; head < shape.heads; ++head) {
                const int group = head / heads_per_group;
                const std::size_t dt_index =
                    ((std::size_t)batch * shape.tokens + token) * shape.heads + head;
                const double dt_value = reference_dt(data.dt[dt_index],
                    data.dt_bias[head], limit_lo, limit_hi);
                const double decay = std::exp(dt_value * -std::exp(data.a_log[head]));
                for (int column = 0; column < shape.columns; ++column) {
                    const std::size_t state_base =
                        (((std::size_t)batch * shape.heads + head) * shape.columns +
                            column) * shape.state_width;
                    const double x_value = data.x[
                        (((std::size_t)batch * shape.tokens + token) * shape.heads +
                            head) * shape.columns + column];
                    double accumulated = 0.0;
                    for (int index = 0; index < shape.state_width; ++index) {
                        const std::size_t gate =
                            ((std::size_t)batch * shape.tokens + token) *
                                shape.groups * shape.state_width +
                            (std::size_t)group * shape.state_width + index;
                        const double updated =
                            state[state_base + index] * decay +
                            dt_value * x_value * data.b[gate];
                        state[state_base + index] = updated;
                        accumulated += updated * data.c[gate];
                    }
                    result.y[(((std::size_t)batch * shape.tokens + token) *
                        shape.heads + head) * shape.columns + column] =
                        accumulated + data.d[head] * x_value;
                }
            }
            const auto batch_state =
                state.begin() + (std::size_t)batch * plane;
            std::copy(batch_state, batch_state + plane, result.steps.begin() +
                ((std::size_t)batch * shape.tokens + token) * plane);
        }
    }
    result.state = state;
    return result;
}

ScanData make_scan_data(const ScanShape& shape, tb::ScalarType dtype,
                        float x_scale, float dt_scale)
{
    require(shape.heads % shape.groups == 0, "test head/group mismatch");
    require(shape.state_width % 32 == 0, "test state width must be 32-aligned");
    ScanData data;
    data.shape = shape;
    const std::int64_t batches = shape.batches, tokens = shape.tokens;
    const std::int64_t heads = shape.heads, columns = shape.columns;
    const std::int64_t groups = shape.groups, width = shape.state_width;
    const std::size_t x_count = (std::size_t)batches * tokens * heads * columns;
    const std::size_t gate_count = (std::size_t)batches * tokens * groups * width;
    data.x = widen(round_to(pattern(x_count, x_scale, 0.7f),
        {batches, tokens, heads, columns}, dtype));
    data.dt = widen(round_to(pattern((std::size_t)batches * tokens * heads, dt_scale, 1.9f),
        {batches, tokens, heads}, dtype));
    data.b = widen(round_to(pattern(gate_count, 0.9f, 4.4f),
        {batches, tokens, groups, width}, dtype));
    data.c = widen(round_to(pattern(gate_count, 0.8f, 5.6f),
        {batches, tokens, groups, width}, dtype));
    data.dt_bias = pattern(heads, 0.4f, 2.3f);
    data.a_log = pattern(heads, 0.6f, 3.1f);
    data.d = pattern(heads, 0.5f, 6.2f);
    return data;
}

std::vector<float> make_state(const ScanShape& shape, float scale)
{
    return pattern((std::size_t)shape.batches * shape.heads * shape.columns *
        shape.state_width, scale, 8.1f);
}

std::vector<float> gather_x(const ScanData& data, int token)
{
    const ScanShape& shape = data.shape;
    std::vector<float> values((std::size_t)shape.batches * shape.heads * shape.columns);
    for (int batch = 0; batch < shape.batches; ++batch) {
        for (int head = 0; head < shape.heads; ++head) {
            for (int column = 0; column < shape.columns; ++column) {
                values[((std::size_t)batch * shape.heads + head) * shape.columns +
                    column] = (float)data.x[
                    (((std::size_t)batch * shape.tokens + token) * shape.heads +
                        head) * shape.columns + column];
            }
        }
    }
    return values;
}

std::vector<float> gather_dt(const ScanData& data, int token)
{
    const ScanShape& shape = data.shape;
    std::vector<float> values((std::size_t)shape.batches * shape.heads);
    for (int batch = 0; batch < shape.batches; ++batch) {
        for (int head = 0; head < shape.heads; ++head) {
            values[(std::size_t)batch * shape.heads + head] =
                (float)data.dt[((std::size_t)batch * shape.tokens + token) *
                    shape.heads + head];
        }
    }
    return values;
}

std::vector<float> gather_gate(const std::vector<double>& source,
                               const ScanData& data, int token)
{
    const ScanShape& shape = data.shape;
    std::vector<float> values((std::size_t)shape.batches * shape.groups *
        shape.state_width);
    for (int batch = 0; batch < shape.batches; ++batch) {
        for (int group = 0; group < shape.groups; ++group) {
            for (int index = 0; index < shape.state_width; ++index) {
                values[((std::size_t)batch * shape.groups + group) *
                    shape.state_width + index] = (float)source[
                    ((std::size_t)batch * shape.tokens + token) * shape.groups *
                        shape.state_width +
                    (std::size_t)group * shape.state_width + index];
            }
        }
    }
    return values;
}

Tensor upload_parameter(const std::vector<float>& values, int count)
{
    return upload(values, {(std::int64_t)count});
}

struct ScanTensors {
    Tensor x;   // [B,T,H,P]
    Tensor dt;  // [B,T,H]
    Tensor b;   // [B,T,G,N]
    Tensor c;   // [B,T,G,N]
};

ScanTensors upload_scan(const ScanData& data, tb::ScalarType dtype)
{
    const ScanShape& shape = data.shape;
    return {
        upload_as(narrow(data.x),
            {shape.batches, shape.tokens, shape.heads, shape.columns}, dtype),
        upload_as(narrow(data.dt),
            {shape.batches, shape.tokens, shape.heads}, dtype),
        upload_as(narrow(data.b),
            {shape.batches, shape.tokens, shape.groups, shape.state_width}, dtype),
        upload_as(narrow(data.c),
            {shape.batches, shape.tokens, shape.groups, shape.state_width}, dtype)};
}

struct ScanParameters {
    Tensor dt_bias;  // [H] f32
    Tensor a_log;    // [H] f32
    Tensor skip;     // [H] f32
};

ScanParameters upload_parameters(const ScanData& data)
{
    const int heads = data.shape.heads;
    return {upload_parameter(data.dt_bias, heads),
        upload_parameter(data.a_log, heads),
        upload_parameter(data.d, heads)};
}

// The reference keeps y as [B,T,H,P]; the tests also need it step-major
// [T][B,H,P] to line up with per-step decode outputs.
template <class Value>
std::vector<Value> step_major(const std::vector<Value>& values,
                              const ScanShape& shape)
{
    std::vector<Value> result(values.size(), Value{});
    for (int batch = 0; batch < shape.batches; ++batch) {
        for (int token = 0; token < shape.tokens; ++token) {
            for (int head = 0; head < shape.heads; ++head) {
                for (int column = 0; column < shape.columns; ++column) {
                    result[(((std::size_t)token * shape.batches + batch) *
                        shape.heads + head) * shape.columns + column] =
                        values[(((std::size_t)batch * shape.tokens + token) *
                            shape.heads + head) * shape.columns + column];
                }
            }
        }
    }
    return result;
}

// Compare decode steps with the double reference, including the in-place state
// evolution and the grouped (G < H) B/C broadcast.
void check_decode(const ScanShape& shape, tb::ScalarType dtype,
                  float x_scale, float dt_scale)
{
    const ScanData data = make_scan_data(shape, dtype, x_scale, dt_scale);
    const std::vector<float> initial_values = make_state(shape, 0.5f);
    const ScanReference reference =
        reference_scan(data, widen(initial_values), 0.0, kUnboundedLimit);

    auto state = upload(initial_values,
        {(std::int64_t)shape.batches, (std::int64_t)shape.heads,
         (std::int64_t)shape.columns, (std::int64_t)shape.state_width});
    const ScanParameters parameters = upload_parameters(data);
    const std::vector<double> reference_steps = step_major(reference.y, shape);
    const std::size_t plane =
        (std::size_t)shape.heads * shape.columns * shape.state_width;
    for (int token = 0; token < shape.tokens; ++token) {
        // Successive calls on the same state buffer must chain exactly like the
        // recurrence: the op owns the in-place update.
        auto x = upload_as(gather_x(data, token),
            {shape.batches, shape.heads, shape.columns}, dtype);
        auto dt = upload_as(gather_dt(data, token),
            {shape.batches, shape.heads}, dtype);
        auto b = upload_as(gather_gate(data.b, data, token),
            {shape.batches, shape.groups, shape.state_width}, dtype);
        auto c = upload_as(gather_gate(data.c, data, token),
            {shape.batches, shape.groups, shape.state_width}, dtype);
        auto y = selective_state_scan_decode_cuda(state, x, dt,
            parameters.dt_bias, parameters.a_log, b, c, parameters.skip, 0.0,
            kUnboundedLimit);
        require(y.sizes().vec() == std::vector<std::int64_t>(
            {shape.batches, shape.heads, shape.columns}), "decode y shape");
        require(y.scalar_type() == tb::kFloat32, "decode y must be f32");
        const std::size_t step = (std::size_t)shape.batches * shape.heads *
            shape.columns;
        close_vector(download(y),
            std::vector<double>(reference_steps.begin() + (std::size_t)token * step,
                reference_steps.begin() + (std::size_t)(token + 1) * step),
            1.0e-5f, 1.0e-4f, "decode y token=" + std::to_string(token));
        std::vector<double> expected_state;
        for (int batch = 0; batch < shape.batches; ++batch) {
            const auto begin = reference.steps.begin() +
                ((std::size_t)batch * shape.tokens + token) * plane;
            expected_state.insert(expected_state.end(), begin, begin + plane);
        }
        close_vector(download(state), expected_state,
            1.0e-5f, 1.0e-4f, "decode state token=" + std::to_string(token));
    }
}

// Prefill must equal the token-by-token recurrence for any T, with and without
// an initial state, and must return the carried final state.
void check_prefill(const ScanShape& shape, tb::ScalarType dtype,
                   bool with_initial_state)
{
    const ScanData data = make_scan_data(shape, dtype, 0.9f, 0.7f);
    const std::vector<float> initial_values =
        with_initial_state ? make_state(shape, 0.4f) : std::vector<float>(
            (std::size_t)shape.batches * shape.heads * shape.columns *
            shape.state_width, 0.0f);
    const ScanReference reference =
        reference_scan(data, widen(initial_values), 0.0, kUnboundedLimit);

    const ScanTensors inputs = upload_scan(data, dtype);
    const ScanParameters parameters = upload_parameters(data);
    Tensor initial_state;
    Tensor initial_copy;
    if (with_initial_state) {
        initial_state = upload(initial_values,
            {(std::int64_t)shape.batches, (std::int64_t)shape.heads,
             (std::int64_t)shape.columns, (std::int64_t)shape.state_width});
        initial_copy = initial_state.clone();
    }
    auto result = selective_state_scan_prefill_cuda(inputs.x, inputs.dt,
        parameters.dt_bias, parameters.a_log, inputs.b, inputs.c, parameters.skip,
        initial_state, 0.0, kUnboundedLimit);
    require(result.size() == 2, "prefill must return y and final_state");
    const Tensor& y = result[0];
    const Tensor& final_state = result[1];
    require(y.sizes().vec() == std::vector<std::int64_t>({shape.batches,
        shape.tokens, shape.heads, shape.columns}), "prefill y shape");
    require(final_state.sizes().vec() == std::vector<std::int64_t>({shape.batches,
        shape.heads, shape.columns, shape.state_width}), "prefill final state shape");
    require(y.scalar_type() == tb::kFloat32 &&
        final_state.scalar_type() == tb::kFloat32, "prefill outputs must be f32");
    close_vector(download(y), reference.y, 1.0e-5f, 1.0e-4f, "prefill y");
    close_vector(download(final_state), reference.state, 1.0e-5f, 1.0e-4f,
        "prefill final state");
    if (with_initial_state) {
        close_vector(download(initial_state), widen(download(initial_copy)),
            0.0f, 0.0f, "prefill must not mutate initial_state");
    }

    // Same inputs through the decode entry point, token by token.
    auto state = with_initial_state
        ? upload(initial_values,
            {(std::int64_t)shape.batches, (std::int64_t)shape.heads,
             (std::int64_t)shape.columns, (std::int64_t)shape.state_width})
        : tb::zeros({(std::int64_t)shape.batches, (std::int64_t)shape.heads,
            (std::int64_t)shape.columns, (std::int64_t)shape.state_width},
            tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCUDA));
    std::vector<double> stepwise_y(reference.y.size(), 0.0);
    const std::size_t step = (std::size_t)shape.batches * shape.heads *
        shape.columns;
    for (int token = 0; token < shape.tokens; ++token) {
        auto step_y = selective_state_scan_decode_cuda(
            state, upload_as(gather_x(data, token),
                {shape.batches, shape.heads, shape.columns}, dtype),
            upload_as(gather_dt(data, token), {shape.batches, shape.heads}, dtype),
            parameters.dt_bias, parameters.a_log,
            upload_as(gather_gate(data.b, data, token),
                {shape.batches, shape.groups, shape.state_width}, dtype),
            upload_as(gather_gate(data.c, data, token),
                {shape.batches, shape.groups, shape.state_width}, dtype),
            parameters.skip, 0.0, kUnboundedLimit);
        const std::vector<float> values = download(step_y);
        for (std::size_t index = 0; index < step; ++index) {
            stepwise_y[(std::size_t)token * step + index] = values[index];
        }
    }
    close_vector(step_major(download(y), shape), stepwise_y, 1.0e-5f, 1.0e-4f,
        "prefill versus stepwise decode y");
    close_vector(download(final_state), widen(download(state)), 1.0e-5f, 1.0e-4f,
        "prefill versus stepwise decode final state");
}

// A finite upper clamp must change the result and must match the clamped
// reference.
void check_dt_clamp()
{
    ScanShape shape;
    shape.batches = 1;
    shape.tokens = 5;
    shape.heads = 4;
    shape.columns = 3;
    shape.state_width = 32;
    shape.groups = 2;
    const ScanData data = make_scan_data(shape, tb::kFloat32, 1.1f, 6.0f);
    const std::vector<float> initial_values = make_state(shape, 0.3f);
    const ScanTensors inputs = upload_scan(data, tb::kFloat32);
    const ScanParameters parameters = upload_parameters(data);
    auto initial_state = upload(initial_values,
        {shape.batches, shape.heads, shape.columns, shape.state_width});

    const double limit_hi = 0.05;
    auto clamped = selective_state_scan_prefill_cuda(inputs.x, inputs.dt,
        parameters.dt_bias, parameters.a_log, inputs.b, inputs.c, parameters.skip,
        initial_state, 0.0, limit_hi);
    auto free_running = selective_state_scan_prefill_cuda(inputs.x, inputs.dt,
        parameters.dt_bias, parameters.a_log, inputs.b, inputs.c, parameters.skip,
        initial_state, 0.0, kUnboundedLimit);
    const ScanReference clamped_reference =
        reference_scan(data, widen(initial_values), 0.0, limit_hi);
    const ScanReference free_reference =
        reference_scan(data, widen(initial_values), 0.0, kUnboundedLimit);
    close_vector(download(clamped[0]), clamped_reference.y, 1.0e-5f, 1.0e-4f,
        "clamped prefill y");
    close_vector(download(clamped[1]), clamped_reference.state, 1.0e-5f, 1.0e-4f,
        "clamped prefill final state");
    close_vector(download(free_running[0]), free_reference.y, 1.0e-5f, 1.0e-4f,
        "unclamped prefill y");
    const float separation = peak_difference(download(clamped[0]),
        widen(download(free_running[0])));
    require(separation > 1.0e-3f,
        "a finite dt_limit_hi must change the scan output, separation=" +
            std::to_string(separation));
}

// ---------------------------------------------------------------------------
// Grouped gated RMSNorm
// ---------------------------------------------------------------------------

// Gate first, then the group mean square (see grouped_gated_rms_norm_cuda).
std::vector<double> reference_gated_norm(const std::vector<double>& x,
                                         const std::vector<double>& gate,
                                         const std::vector<double>& weight,
                                         int rows, int width, int group,
                                         double eps)
{
    std::vector<double> output(x.size(), 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int index = 0; index < width / group; ++index) {
            const int base = row * width + index * group;
            std::vector<double> gated(group);
            double square = 0.0;
            for (int column = 0; column < group; ++column) {
                const double z = gate[base + column];
                gated[column] = x[base + column] * (z / (1.0 + std::exp(-z)));
                square += gated[column] * gated[column];
            }
            const double inverse = 1.0 / std::sqrt(square / group + eps);
            for (int column = 0; column < group; ++column) {
                output[base + column] = weight[index * group + column] *
                    gated[column] * inverse;
            }
        }
    }
    return output;
}

// The order the operator must *not* implement: normalize first, gate after.
std::vector<double> reference_norm_then_gate(const std::vector<double>& x,
                                             const std::vector<double>& gate,
                                             const std::vector<double>& weight,
                                             int rows, int width, int group,
                                             double eps)
{
    std::vector<double> output(x.size(), 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int index = 0; index < width / group; ++index) {
            const int base = row * width + index * group;
            double square = 0.0;
            for (int column = 0; column < group; ++column) {
                square += x[base + column] * x[base + column];
            }
            const double inverse = 1.0 / std::sqrt(square / group + eps);
            for (int column = 0; column < group; ++column) {
                const double z = gate[base + column];
                output[base + column] = weight[index * group + column] *
                    x[base + column] * inverse * (z / (1.0 + std::exp(-z)));
            }
        }
    }
    return output;
}

void check_gated_norm(tb::ScalarType x_dtype, tb::ScalarType gate_dtype)
{
    const int rows = 5, width = 16, group = 4;
    const double eps = 1.0e-6;
    // A 10x amplitude gap between x and the gate is what makes a reversed gate
    // order visible: the two orders differ by a gate-dependent normalizer.
    std::vector<float> x_values = pattern((std::size_t)rows * width, 1.0f, 1.3f);
    std::vector<float> gate_values = pattern((std::size_t)rows * width, 0.1f, 9.7f);
    std::vector<float> weight_values = pattern((std::size_t)width, 0.5f, 2.9f);
    for (int index = 0; index < width; ++index) {
        weight_values[index] += 1.0f;
    }
    const std::vector<float> rounded_x =
        round_to(x_values, {rows, width}, x_dtype);
    const std::vector<float> rounded_gate =
        round_to(gate_values, {rows, width}, gate_dtype);

    auto x = upload_as(x_values, {rows, width}, x_dtype);
    auto gate = upload_as(gate_values, {rows, width}, gate_dtype);
    auto weight = upload(weight_values, {width});
    auto y = grouped_gated_rms_norm_cuda(x, gate, weight, group, eps);
    require(y.sizes().vec() == std::vector<std::int64_t>({rows, width}),
        "gated norm output shape");
    require(y.scalar_type() == x_dtype, "gated norm output must keep the x dtype");

    const std::vector<double> expected = reference_gated_norm(
        widen(rounded_x), widen(rounded_gate), widen(weight_values),
        rows, width, group, eps);
    const bool wide_output = x_dtype != tb::kFloat32;
    close_vector(download(y), expected, wide_output ? 5.0e-3f : 1.0e-5f,
        wide_output ? 1.0e-2f : 1.0e-4f, "gated norm");

    // Reversing the order (normalize first) must be a different function.
    const std::vector<double> reversed = reference_norm_then_gate(
        widen(rounded_x), widen(rounded_gate), widen(weight_values),
        rows, width, group, eps);
    const float separation = peak_difference(download(y), reversed);
    require(separation > 1.0e-2f,
        "gated norm must apply the gate before the group mean square, "
        "separation=" + std::to_string(separation));
}

// ---------------------------------------------------------------------------
// relu^2
// ---------------------------------------------------------------------------

void check_relu_squared(tb::ScalarType dtype)
{
    const std::vector<float> values = {
        -8.0f, -3.5f, -1.0f, -0.25f, 0.0f, 0.125f, 1.0f, 2.5f, 7.75f,
        -12.5f, 4.0f};
    const std::int64_t count = (std::int64_t)values.size();
    const std::vector<float> rounded = round_to(values, {count}, dtype);
    auto x = upload_as(values, {count}, dtype);
    auto y = relu_squared_cuda(x);
    require(y.sizes().vec() == std::vector<std::int64_t>({count}),
        "relu_squared output shape");
    require(y.scalar_type() == dtype, "relu_squared must keep the input dtype");
    std::vector<double> expected(rounded.size(), 0.0);
    for (std::size_t index = 0; index < rounded.size(); ++index) {
        const double relu = std::max((double)rounded[index], 0.0);
        expected[index] = relu * relu;
    }
    const bool wide_output = dtype != tb::kFloat32;
    close_vector(download(y), expected, wide_output ? 2.0e-2f : 1.0e-6f,
        wide_output ? 2.0e-2f : 1.0e-6f, "relu_squared");
}

}  // namespace

int main()
{
    int devices = 0;
    MFQ_NATIVE_CUDA_CHECK(hipGetDeviceCount(&devices));
    if (devices == 0) {
        std::cout << "recurrent mixer ops: no CUDA device\n";
        return 77;
    }

    int cases = 0;
    // Decode: grouped B/C (G < H), several successive in-place steps.
    for (auto dtype : {tb::kFloat32, tb::kBFloat16}) {
        ScanShape shape;
        shape.batches = 2;
        shape.tokens = 3;
        shape.heads = 8;
        shape.columns = 4;
        shape.state_width = 32;
        shape.groups = 2;
        check_decode(shape, dtype, 1.0f, 0.8f);
        ++cases;
        ScanShape narrow;
        narrow.batches = 1;
        narrow.tokens = 2;
        narrow.heads = 4;
        narrow.columns = 3;
        narrow.state_width = 64;
        narrow.groups = 1;
        check_decode(narrow, dtype, 1.4f, 1.6f);
        ++cases;
    }
    // Prefill: T = 1, T = 257 (not a chunk multiple), B = 2, with/without an
    // initial state, checked against both the reference and stepwise decode.
    for (bool with_initial : {false, true}) {
        ScanShape single;
        single.batches = 2;
        single.tokens = 1;
        single.heads = 4;
        single.columns = 4;
        single.state_width = 32;
        single.groups = 2;
        check_prefill(single, tb::kFloat32, with_initial);
        ++cases;
        ScanShape long_form = single;
        long_form.tokens = 257;
        check_prefill(long_form, tb::kFloat32, with_initial);
        ++cases;
        ScanShape bf16_form = long_form;
        bf16_form.tokens = 65;
        check_prefill(bf16_form, tb::kBFloat16, with_initial);
        ++cases;
    }
    {
        // A long prompt (1024 tokens) must stay exact against the reference
        // even though it is far beyond any internal chunking.
        ScanShape long_form;
        long_form.batches = 1;
        long_form.tokens = 1024;
        long_form.heads = 4;
        long_form.columns = 4;
        long_form.state_width = 32;
        long_form.groups = 2;
        check_prefill(long_form, tb::kFloat32, true);
        ++cases;
    }
    check_dt_clamp();
    ++cases;
    check_gated_norm(tb::kFloat32, tb::kFloat32);
    ++cases;
    check_gated_norm(tb::kFloat32, tb::kBFloat16);
    ++cases;
    check_gated_norm(tb::kBFloat16, tb::kBFloat16);
    ++cases;
    check_gated_norm(tb::kFloat16, tb::kFloat16);
    ++cases;
    for (auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16}) {
        check_relu_squared(dtype);
        ++cases;
    }

    std::cout << "recurrent_mixer_ops cases=" << cases << " PASS\n";
    return 0;
}
