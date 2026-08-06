#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> nvq1_l_assign_cuda(
    torch::Tensor xgroup,
    torch::Tensor wgroup,
    torch::Tensor group_anchor,
    torch::Tensor codebook,
    int64_t groups_per_row,
    int64_t valid_last,
    int64_t sub_bits,
    double delta);
std::vector<torch::Tensor> nvq_search_cuda(
    torch::Tensor xgroup,
    torch::Tensor wgroup,
    torch::Tensor codebook,
    int64_t groups_per_row,
    int64_t valid_last,
    int64_t vector_size,
    int64_t search_steps,
    double qmax);
torch::Tensor nvq_reassign_cuda(
    torch::Tensor xgroup,
    torch::Tensor wgroup,
    torch::Tensor scale,
    torch::Tensor codebook,
    int64_t groups_per_row,
    int64_t valid_last,
    int64_t vector_size);
std::vector<torch::Tensor> nepq0_s_assign_cuda(
    torch::Tensor value,
    torch::Tensor initial_anchor,
    torch::Tensor scale_lut,
    torch::Tensor first_tables,
    torch::Tensor second_tables);
std::vector<torch::Tensor> npq0_s_assign_cuda(
    torch::Tensor value,
    torch::Tensor objective_weight,
    torch::Tensor initial_anchor,
    torch::Tensor scale_lut,
    torch::Tensor first_codebooks,
    torch::Tensor second_codebooks,
    int64_t valid_width,
    int64_t refine_steps);
std::vector<torch::Tensor> npq0_l_assign_cuda(
    torch::Tensor value,
    torch::Tensor objective_weight,
    torch::Tensor initial_anchor,
    torch::Tensor scale_lut,
    torch::Tensor first_codebooks,
    torch::Tensor second_codebooks,
    int64_t valid_width,
    int64_t refine_steps);
std::vector<torch::Tensor> nvq2j_assign_cuda(
    torch::Tensor value,
    torch::Tensor objective_weight,
    torch::Tensor initial_anchor,
    torch::Tensor scale_lut,
    torch::Tensor bank_for_state,
    torch::Tensor codebooks,
    int64_t valid_width,
    int64_t refine_steps);
std::vector<torch::Tensor> nvq3j_assign_cuda(
    torch::Tensor value,
    torch::Tensor objective_weight,
    torch::Tensor initial_anchor,
    torch::Tensor scale_lut,
    torch::Tensor bank_for_state,
    torch::Tensor codebooks,
    int64_t valid_width,
    int64_t refine_steps);
std::vector<torch::Tensor> nvq2j_search_banks_cuda(
    torch::Tensor xgroup,
    torch::Tensor wgroup,
    torch::Tensor codebooks,
    torch::Tensor bank_qmax,
    int64_t groups_per_row,
    int64_t valid_last,
    int64_t search_steps);
std::vector<torch::Tensor> nint_make_qkx3_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t nmax,
    double rmin,
    double rdelta,
    int64_t nstep);
std::vector<torch::Tensor> nint_make_qp_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t nmax);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "nvq1_l_assign",
        &nvq1_l_assign_cuda,
        "Exact NVQ1-L fixed-anchor group assignment (CUDA)");
    m.def("nvq_search", &nvq_search_cuda, "NVQ floating group-scale search (CUDA)");
    m.def("nvq_reassign", &nvq_reassign_cuda, "NVQ fixed-scale code assignment (CUDA)");
    m.def(
        "nepq0_s_assign",
        &nepq0_s_assign_cuda,
        "NEPQ0-S 256-bank assignment and one-step anchor refit (CUDA)");
    m.def(
        "npq0_s_assign",
        &npq0_s_assign_cuda,
        "NPQ0-S fixed-table assignment and anchor refit (CUDA)");
    m.def(
        "npq0_l_assign",
        &npq0_l_assign_cuda,
        "NPQ0-L fixed-table assignment and anchor refit (CUDA)");
    m.def(
        "nvq2j_assign",
        &nvq2j_assign_cuda,
        "NVQ2J fixed-table assignment and anchor refit (CUDA)");
    m.def(
        "nvq3j_assign",
        &nvq3j_assign_cuda,
        "NVQ3J fixed-table assignment and anchor refit (CUDA)");
    m.def(
        "nvq2j_search_banks",
        &nvq2j_search_banks_cuda,
        "NVQ2J fused four-bank floating-scale search (CUDA)");
    m.def(
        "nint_make_qkx3",
        &nint_make_qkx3_cuda,
        "NINT weighted affine group search (CUDA)");
    m.def(
        "nint_make_qp",
        &nint_make_qp_cuda,
        "NINT weighted neuron-scale quantization (CUDA)");
    m.attr("niq_search") = m.attr("nvq_search");
    m.attr("niq_reassign") = m.attr("nvq_reassign");
}
