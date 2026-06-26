// Experimental OpenMX scfout -> symmetrized H.dat/S.dat writer.
//
// Build inside OpenMX source with OPENMX_ANALYSIS_BUILD so read_scfout.h and
// OpenMX globals are available. Without that macro, the same source provides a
// --mock-input path used by release tests for the symmetry core.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef OPENMX_ANALYSIS_BUILD
#include <mpi.h>
extern "C" {
void read_scfout(char* argv[]);
void free_scfout(void);
extern int atomnum;
extern int SpinP_switch;
extern int* Total_NumOrbs;
extern int* FNAN;
extern int** natn;
extern int** ncn;
extern int** atv_ijk;
extern double *****Hks;
extern double *****iHks;
extern double ****OLP;
}
#endif

namespace {

constexpr double kHartreeToEv = 27.2113862;

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string format_seconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << seconds;
    return out.str();
}

void log_step_start(int step, int total, const std::string& label) {
    std::cout << "[" << step << "/" << total << "] " << label << " ..." << std::endl;
}

void log_step_done(int step, int total, const std::string& label, double seconds) {
    std::cout << "[" << step << "/" << total << "] " << label << " done in " << format_seconds(seconds) << " s" << std::endl;
}

void log_step_skip(int step, int total, const std::string& label) {
    std::cout << "[" << step << "/" << total << "] " << label << " skipped" << std::endl;
}

class ScopedStdoutSilencer {
   public:
    explicit ScopedStdoutSilencer(bool enabled) : enabled_(enabled) {
        if (!enabled_) return;
        std::cout.flush();
        std::fflush(stdout);
        saved_fd_ = dup(STDOUT_FILENO);
        int null_fd = open("/dev/null", O_WRONLY);
        if (saved_fd_ >= 0 && null_fd >= 0) {
            dup2(null_fd, STDOUT_FILENO);
        }
        if (null_fd >= 0) close(null_fd);
    }

    ~ScopedStdoutSilencer() {
        if (!enabled_) return;
        std::fflush(stdout);
        if (saved_fd_ >= 0) {
            dup2(saved_fd_, STDOUT_FILENO);
            close(saved_fd_);
        }
    }

    ScopedStdoutSilencer(const ScopedStdoutSilencer&) = delete;
    ScopedStdoutSilencer& operator=(const ScopedStdoutSilencer&) = delete;

   private:
    bool enabled_ = false;
    int saved_fd_ = -1;
};

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    const Json& at(const std::string& key) const {
        auto it = object.find(key);
        if (it == object.end()) {
            throw std::runtime_error("Missing JSON key: " + key);
        }
        return it->second;
    }

    bool contains(const std::string& key) const {
        return object.find(key) != object.end();
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string text) : text_(std::move(text)) {}

    Json parse() {
        Json value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            throw std::runtime_error("Unexpected trailing JSON data.");
        }
        return value;
    }

private:
    const std::string text_;
    std::size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    char peek() {
        skip_ws();
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Unexpected end of JSON.");
        }
        return text_[pos_];
    }

    char get() {
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Unexpected end of JSON.");
        }
        return text_[pos_++];
    }

    Json parse_value() {
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            Json value;
            value.type = Json::Type::String;
            value.string = parse_string();
            return value;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        if (text_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            Json value;
            value.type = Json::Type::Bool;
            value.boolean = true;
            return value;
        }
        if (text_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            Json value;
            value.type = Json::Type::Bool;
            value.boolean = false;
            return value;
        }
        if (text_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return Json{};
        }
        throw std::runtime_error("Invalid JSON value.");
    }

    Json parse_object() {
        Json value;
        value.type = Json::Type::Object;
        get();  // {
        skip_ws();
        if (peek_no_ws() == '}') {
            get();
            return value;
        }
        while (true) {
            if (peek() != '"') {
                throw std::runtime_error("Expected JSON object key.");
            }
            std::string key = parse_string();
            if (peek() != ':') {
                throw std::runtime_error("Expected ':' after JSON object key.");
            }
            get();
            value.object.emplace(std::move(key), parse_value());
            char c = peek();
            if (c == '}') {
                get();
                return value;
            }
            if (c != ',') {
                throw std::runtime_error("Expected ',' or '}' in JSON object.");
            }
            get();
        }
    }

    Json parse_array() {
        Json value;
        value.type = Json::Type::Array;
        get();  // [
        skip_ws();
        if (peek_no_ws() == ']') {
            get();
            return value;
        }
        while (true) {
            value.array.push_back(parse_value());
            char c = peek();
            if (c == ']') {
                get();
                return value;
            }
            if (c != ',') {
                throw std::runtime_error("Expected ',' or ']' in JSON array.");
            }
            get();
        }
    }

    char peek_no_ws() {
        skip_ws();
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Unexpected end of JSON.");
        }
        return text_[pos_];
    }

    std::string parse_string() {
        if (get() != '"') {
            throw std::runtime_error("Expected JSON string.");
        }
        std::string out;
        while (pos_ < text_.size()) {
            char c = get();
            if (c == '"') {
                return out;
            }
            if (c == '\\') {
                char escaped = get();
                if (escaped == '"' || escaped == '\\' || escaped == '/') out.push_back(escaped);
                else if (escaped == 'b') out.push_back('\b');
                else if (escaped == 'f') out.push_back('\f');
                else if (escaped == 'n') out.push_back('\n');
                else if (escaped == 'r') out.push_back('\r');
                else if (escaped == 't') out.push_back('\t');
                else throw std::runtime_error("Unsupported JSON string escape.");
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated JSON string.");
    }

    Json parse_number() {
        skip_ws();
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        double number = std::strtod(begin, &end);
        if (end == begin) {
            throw std::runtime_error("Invalid JSON number.");
        }
        pos_ += static_cast<std::size_t>(end - begin);
        Json value;
        value.type = Json::Type::Number;
        value.number = number;
        return value;
    }
};

Json read_json_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open JSON file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return JsonParser(buffer.str()).parse();
}

int as_int(const Json& value) {
    if (value.type != Json::Type::Number) throw std::runtime_error("Expected JSON number.");
    return static_cast<int>(std::llround(value.number));
}

double as_double(const Json& value) {
    if (value.type != Json::Type::Number) throw std::runtime_error("Expected JSON number.");
    return value.number;
}

bool as_bool(const Json& value) {
    if (value.type != Json::Type::Bool) throw std::runtime_error("Expected JSON bool.");
    return value.boolean;
}

std::vector<int> as_int_vector(const Json& value) {
    if (value.type != Json::Type::Array) throw std::runtime_error("Expected JSON array.");
    std::vector<int> out;
    out.reserve(value.array.size());
    for (const auto& item : value.array) out.push_back(as_int(item));
    return out;
}

std::array<int, 3> as_int3(const Json& value) {
    auto values = as_int_vector(value);
    if (values.size() != 3) throw std::runtime_error("Expected integer triplet.");
    return {values[0], values[1], values[2]};
}

std::array<std::array<double, 3>, 3> as_real33(const Json& value) {
    if (value.type != Json::Type::Array || value.array.size() != 3) {
        throw std::runtime_error("Expected 3x3 JSON array.");
    }
    std::array<std::array<double, 3>, 3> out{};
    for (std::size_t i = 0; i < 3; ++i) {
        if (value.array[i].type != Json::Type::Array || value.array[i].array.size() != 3) {
            throw std::runtime_error("Expected 3x3 JSON array row.");
        }
        for (std::size_t j = 0; j < 3; ++j) out[i][j] = as_double(value.array[i].array[j]);
    }
    return out;
}

std::complex<double> as_complex_pair(const Json& value) {
    if (value.type != Json::Type::Array || value.array.size() != 2) {
        throw std::runtime_error("Expected complex pair [real, imag].");
    }
    return {as_double(value.array[0]), as_double(value.array[1])};
}

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<std::complex<double>> data;

    Matrix() = default;
    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<std::size_t>(r * c), {0.0, 0.0}) {}

    std::complex<double>& at(int r, int c) {
        return data[static_cast<std::size_t>(r * cols + c)];
    }

    const std::complex<double>& at(int r, int c) const {
        return data[static_cast<std::size_t>(r * cols + c)];
    }
};

struct NpzEntry {
    std::string name;
    std::uint32_t crc = 0;
    std::uint64_t compressed_size = 0;
    std::uint64_t uncompressed_size = 0;
    std::uint64_t local_header_offset = 0;
};

Matrix as_complex_matrix(const Json& value) {
    if (value.type != Json::Type::Array) throw std::runtime_error("Expected matrix array.");
    int rows = static_cast<int>(value.array.size());
    int cols = rows == 0 ? 0 : static_cast<int>(value.array[0].array.size());
    Matrix matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        if (value.array[static_cast<std::size_t>(i)].type != Json::Type::Array ||
            static_cast<int>(value.array[static_cast<std::size_t>(i)].array.size()) != cols) {
            throw std::runtime_error("Ragged complex matrix JSON.");
        }
        for (int j = 0; j < cols; ++j) {
            matrix.at(i, j) = as_complex_pair(value.array[static_cast<std::size_t>(i)].array[static_cast<std::size_t>(j)]);
        }
    }
    return matrix;
}

Matrix dagger(const Matrix& matrix) {
    Matrix out(matrix.cols, matrix.rows);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            out.at(j, i) = std::conj(matrix.at(i, j));
        }
    }
    return out;
}

Matrix add(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) throw std::runtime_error("Matrix shape mismatch in add.");
    Matrix out(a.rows, a.cols);
    for (std::size_t i = 0; i < out.data.size(); ++i) out.data[i] = a.data[i] + b.data[i];
    return out;
}

Matrix scale(const Matrix& a, double factor) {
    Matrix out(a.rows, a.cols);
    for (std::size_t i = 0; i < out.data.size(); ++i) out.data[i] = a.data[i] * factor;
    return out;
}

void add_inplace(Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) throw std::runtime_error("Matrix shape mismatch in add_inplace.");
    for (std::size_t i = 0; i < a.data.size(); ++i) a.data[i] += b.data[i];
}

Matrix matmul(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) throw std::runtime_error("Matrix shape mismatch in matmul.");
    Matrix out(a.rows, b.cols);
    for (int i = 0; i < a.rows; ++i) {
        for (int k = 0; k < a.cols; ++k) {
            const auto av = a.at(i, k);
            if (std::abs(av) == 0.0) continue;
            for (int j = 0; j < b.cols; ++j) {
                out.at(i, j) += av * b.at(k, j);
            }
        }
    }
    return out;
}

Matrix transform_block(const Matrix& left, const Matrix& block, const Matrix& right_h) {
    return matmul(matmul(left, block), right_h);
}

Matrix conjugate_matrix(const Matrix& matrix) {
    Matrix out(matrix.rows, matrix.cols);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) out.at(i, j) = std::conj(matrix.at(i, j));
    }
    return out;
}

Matrix transform_block(const Matrix& left, const Matrix& block, const Matrix& right_h, bool antiunitary) {
    if (!antiunitary) return transform_block(left, block, right_h);
    Matrix conj_block = conjugate_matrix(block);
    return transform_block(left, conj_block, right_h);
}

struct Key {
    int rx = 0, ry = 0, rz = 0;
    int atom_i = 0, atom_j = 0;

    bool operator<(const Key& other) const {
        return std::tie(rx, ry, rz, atom_i, atom_j) < std::tie(other.rx, other.ry, other.rz, other.atom_i, other.atom_j);
    }

    bool operator==(const Key& other) const {
        return rx == other.rx && ry == other.ry && rz == other.rz && atom_i == other.atom_i && atom_j == other.atom_j;
    }
};

using BlockMap = std::map<Key, Matrix>;

Key mate_key(const Key& key) {
    return {-key.rx, -key.ry, -key.rz, key.atom_j, key.atom_i};
}

Key canonical_key(const Key& key) {
    Key mate = mate_key(key);
    return (mate < key) ? mate : key;
}

struct Operation {
    int index = 0;
    bool antiunitary = false;
    std::array<std::array<double, 3>, 3> rotation_frac{};
    std::vector<int> atom_mappings;
    std::vector<std::array<int, 3>> image_shifts;
    std::vector<Matrix> orbital_blocks;
    std::vector<Matrix> orbital_blocks_dagger;
};

struct SymmetryData {
    bool spinful = false;
    int natoms = 0;
    int nwann = 0;
    int nwann_spinless = 0;
    std::vector<int> atom_orbital_counts;
    std::vector<int> atom_offsets;
    std::vector<Operation> operations;
};

struct MatrixSet {
    bool spinful = false;
    std::vector<int> atom_orbital_counts;
    std::vector<int> atom_offsets;
    int nwann = 0;
    int nwann_spinless = 0;
    BlockMap h_blocks;
    BlockMap s_blocks;
};

SymmetryData load_symmetry(const std::filesystem::path& path) {
    Json root = read_json_file(path);
    if (root.at("schema").string != "openmx_hs_symmetry_transports/v1") {
        throw std::runtime_error("Unsupported symmetry transport schema.");
    }
    SymmetryData symmetry;
    symmetry.spinful = as_bool(root.at("spinful"));
    symmetry.natoms = as_int(root.at("natoms"));
    symmetry.nwann = as_int(root.at("nwann"));
    symmetry.nwann_spinless = as_int(root.at("nwann_spinless"));
    symmetry.atom_orbital_counts = as_int_vector(root.at("atom_orbital_counts"));
    symmetry.atom_offsets = as_int_vector(root.at("atom_offsets"));
    if (static_cast<int>(symmetry.atom_orbital_counts.size()) != symmetry.natoms) {
        throw std::runtime_error("atom_orbital_counts length does not match natoms.");
    }
    const Json& ops = root.at("operations");
    if (ops.type != Json::Type::Array) throw std::runtime_error("operations must be an array.");
    for (const auto& item : ops.array) {
        Operation op;
        op.index = as_int(item.at("index"));
        op.antiunitary = item.contains("antiunitary") ? as_bool(item.at("antiunitary")) : false;
        op.rotation_frac = as_real33(item.at("rotation_frac"));
        op.atom_mappings = as_int_vector(item.at("atom_mappings"));
        const Json& shifts = item.at("image_shifts");
        for (const auto& shift : shifts.array) op.image_shifts.push_back(as_int3(shift));
        op.orbital_blocks.resize(static_cast<std::size_t>(symmetry.natoms));
        op.orbital_blocks_dagger.resize(static_cast<std::size_t>(symmetry.natoms));
        const Json& blocks = item.at("orbital_blocks");
        for (int atom = 0; atom < symmetry.natoms; ++atom) {
            Matrix block = as_complex_matrix(blocks.at(std::to_string(atom)));
            op.orbital_blocks[static_cast<std::size_t>(atom)] = block;
            op.orbital_blocks_dagger[static_cast<std::size_t>(atom)] = dagger(block);
        }
        symmetry.operations.push_back(std::move(op));
    }
    return symmetry;
}

std::vector<int> offsets_from_counts(const std::vector<int>& counts) {
    std::vector<int> offsets(counts.size(), 0);
    int offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        offsets[i] = offset;
        offset += counts[i];
    }
    return offsets;
}

int total_spinless_orbitals(const std::vector<int>& counts) {
    int total = 0;
    for (int count : counts) total += count;
    return total;
}

void add_sparse_block(BlockMap& blocks, const Json& item, const std::vector<int>& atom_counts, bool spinful) {
    Key key;
    auto r = as_int3(item.at("r"));
    key.rx = r[0];
    key.ry = r[1];
    key.rz = r[2];
    key.atom_i = as_int(item.at("atom_i"));
    key.atom_j = as_int(item.at("atom_j"));
    int spin_factor = spinful ? 2 : 1;
    int rows_dim = spin_factor * atom_counts[static_cast<std::size_t>(key.atom_i)];
    int cols_dim = spin_factor * atom_counts[static_cast<std::size_t>(key.atom_j)];
    auto [it, inserted] = blocks.emplace(key, Matrix(rows_dim, cols_dim));
    Matrix& block = it->second;
    auto rows = as_int_vector(item.at("rows"));
    auto cols = as_int_vector(item.at("cols"));
    const Json& values = item.at("values");
    if (rows.size() != cols.size() || rows.size() != values.array.size()) {
        throw std::runtime_error("Mock sparse block rows/cols/values lengths differ.");
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        block.at(rows[i], cols[i]) += as_complex_pair(values.array[i]);
    }
}

MatrixSet load_mock_input(const std::filesystem::path& path) {
    Json root = read_json_file(path);
    MatrixSet data;
    data.spinful = as_bool(root.at("spinful"));
    data.atom_orbital_counts = as_int_vector(root.at("atom_orbital_counts"));
    data.atom_offsets = offsets_from_counts(data.atom_orbital_counts);
    data.nwann_spinless = total_spinless_orbitals(data.atom_orbital_counts);
    data.nwann = data.spinful ? 2 * data.nwann_spinless : data.nwann_spinless;
    for (const auto& item : root.at("h_blocks").array) add_sparse_block(data.h_blocks, item, data.atom_orbital_counts, data.spinful);
    for (const auto& item : root.at("s_blocks").array) add_sparse_block(data.s_blocks, item, data.atom_orbital_counts, data.spinful);
    return data;
}

void add_dense_value(BlockMap& blocks, const Key& key, int local_row, int local_col, std::complex<double> value, const std::vector<int>& atom_counts, bool spinful) {
    int spin_factor = spinful ? 2 : 1;
    int rows_dim = spin_factor * atom_counts[static_cast<std::size_t>(key.atom_i)];
    int cols_dim = spin_factor * atom_counts[static_cast<std::size_t>(key.atom_j)];
    auto [it, inserted] = blocks.try_emplace(key, rows_dim, cols_dim);
    it->second.at(local_row, local_col) += value;
}

Matrix& get_or_create_block(BlockMap& blocks, const Key& key, int rows_dim, int cols_dim) {
    auto [it, inserted] = blocks.try_emplace(key, rows_dim, cols_dim);
    Matrix& block = it->second;
    if (block.rows != rows_dim || block.cols != cols_dim) {
        throw std::runtime_error("Repeated OpenMX block key has inconsistent dense shape.");
    }
    return block;
}

void merge_block_maps(BlockMap& dest, BlockMap& src) {
    for (auto& [key, block] : src) {
        auto it = dest.find(key);
        if (it == dest.end()) {
            dest.emplace(key, std::move(block));
        } else {
            add_inplace(it->second, block);
        }
    }
    src.clear();
}

#ifdef OPENMX_ANALYSIS_BUILD
struct OpenMxNeighborTask {
    int ct_AN = 0;
    int h_AN = 0;
    int atom_i = 0;
    int atom_j = 0;
    int orb_i = 0;
    int orb_j = 0;
    Key rkey;
    Key down_up_key;
};

class MpiSession {
public:
    MpiSession(int* argc, char*** argv) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(argc, argv);
            owns_mpi_ = true;
        }
    }

    ~MpiSession() {
        if (!owns_mpi_) return;
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) MPI_Finalize();
    }

    MpiSession(const MpiSession&) = delete;
    MpiSession& operator=(const MpiSession&) = delete;

private:
    bool owns_mpi_ = false;
};

void accumulate_openmx_neighbor_task(const OpenMxNeighborTask& task, int spin_switch, BlockMap& h_blocks, BlockMap& s_blocks) {
    int ct_AN = task.ct_AN;
    int h_AN = task.h_AN;
    int orb_i = task.orb_i;
    int orb_j = task.orb_j;
    if (spin_switch == 0) {
        Matrix& h_block = get_or_create_block(h_blocks, task.rkey, orb_i, orb_j);
        Matrix& s_block = get_or_create_block(s_blocks, task.rkey, orb_i, orb_j);
        for (int i = 0; i < orb_i; ++i) {
            std::size_t row_offset = static_cast<std::size_t>(i * orb_j);
            double* h0 = Hks[0][ct_AN][h_AN][i];
            double* s0 = OLP[ct_AN][h_AN][i];
            for (int j = 0; j < orb_j; ++j) {
                std::size_t idx = row_offset + static_cast<std::size_t>(j);
                h_block.data[idx] += std::complex<double>(h0[j] * kHartreeToEv, 0.0);
                s_block.data[idx] += std::complex<double>(s0[j], 0.0);
            }
        }
        return;
    }

    int rows_dim = 2 * orb_i;
    int cols_dim = 2 * orb_j;
    Matrix& h_block = get_or_create_block(h_blocks, task.rkey, rows_dim, cols_dim);
    Matrix& s_block = get_or_create_block(s_blocks, task.rkey, rows_dim, cols_dim);
    if (spin_switch == 1) {
        for (int i = 0; i < orb_i; ++i) {
            double* h_up = Hks[0][ct_AN][h_AN][i];
            double* h_down = Hks[1][ct_AN][h_AN][i];
            double* s0 = OLP[ct_AN][h_AN][i];
            for (int j = 0; j < orb_j; ++j) {
                h_block.at(i, j) += std::complex<double>(h_up[j] * kHartreeToEv, 0.0);
                h_block.at(orb_i + i, orb_j + j) += std::complex<double>(h_down[j] * kHartreeToEv, 0.0);
                s_block.at(i, j) += std::complex<double>(s0[j], 0.0);
                s_block.at(orb_i + i, orb_j + j) += std::complex<double>(s0[j], 0.0);
            }
        }
        return;
    }

    Matrix& h_down_up_block = get_or_create_block(h_blocks, task.down_up_key, 2 * orb_j, 2 * orb_i);
    for (int i = 0; i < orb_i; ++i) {
        double* h0 = Hks[0][ct_AN][h_AN][i];
        double* ih0 = iHks[0][ct_AN][h_AN][i];
        double* h1 = Hks[1][ct_AN][h_AN][i];
        double* ih1 = iHks[1][ct_AN][h_AN][i];
        double* h2 = Hks[2][ct_AN][h_AN][i];
        double* ih2 = iHks[2][ct_AN][h_AN][i];
        double* h3 = Hks[3][ct_AN][h_AN][i];
        double* s0 = OLP[ct_AN][h_AN][i];
        for (int j = 0; j < orb_j; ++j) {
            int row_up = i;
            int row_down = orb_i + i;
            int col_up = j;
            int col_down = orb_j + j;
            h_block.at(row_up, col_up) += std::complex<double>(h0[j] * kHartreeToEv, ih0[j] * kHartreeToEv);
            h_block.at(row_up, col_down) += std::complex<double>(h2[j] * kHartreeToEv, (h3[j] + ih2[j]) * kHartreeToEv);
            h_down_up_block.at(orb_j + j, i) += std::complex<double>(h2[j] * kHartreeToEv, (-h3[j] - ih2[j]) * kHartreeToEv);
            h_block.at(row_down, col_down) += std::complex<double>(h1[j] * kHartreeToEv, ih1[j] * kHartreeToEv);
            s_block.at(row_up, col_up) += std::complex<double>(s0[j], 0.0);
            s_block.at(row_down, col_down) += std::complex<double>(s0[j], 0.0);
        }
    }
}

MatrixSet load_openmx_scfout(int argc, char** argv, std::map<std::string, double>* timings) {
    if (argc < 2) throw std::runtime_error("Missing OpenMX .scfout path.");
    MpiSession mpi(&argc, &argv);
    auto t0 = Clock::now();
    read_scfout(argv);
    if (timings != nullptr) (*timings)["read_scfout_s"] = seconds_since(t0);
    t0 = Clock::now();
    MatrixSet data;
    if (SpinP_switch != 0 && SpinP_switch != 1 && SpinP_switch != 3) {
        throw std::runtime_error("Only SpinP_switch==0, SpinP_switch==1, and SpinP_switch==3 are supported.");
    }
    data.spinful = SpinP_switch == 1 || SpinP_switch == 3;
    data.atom_orbital_counts.reserve(static_cast<std::size_t>(atomnum));
    for (int atom = 1; atom <= atomnum; ++atom) data.atom_orbital_counts.push_back(Total_NumOrbs[atom]);
    data.atom_offsets = offsets_from_counts(data.atom_orbital_counts);
    data.nwann_spinless = total_spinless_orbitals(data.atom_orbital_counts);
    data.nwann = data.spinful ? 2 * data.nwann_spinless : data.nwann_spinless;

    std::vector<OpenMxNeighborTask> tasks;
    std::size_t task_count = 0;
    for (int ct_AN = 1; ct_AN <= atomnum; ++ct_AN) task_count += static_cast<std::size_t>(FNAN[ct_AN] + 1);
    tasks.reserve(task_count);
    for (int ct_AN = 1; ct_AN <= atomnum; ++ct_AN) {
        int atom_i = ct_AN - 1;
        int orb_i = Total_NumOrbs[ct_AN];
        for (int h_AN = 0; h_AN <= FNAN[ct_AN]; ++h_AN) {
            int gh_AN = natn[ct_AN][h_AN];
            int atom_j = gh_AN - 1;
            int rn = ncn[ct_AN][h_AN];
            Key rkey{atv_ijk[rn][1], atv_ijk[rn][2], atv_ijk[rn][3], atom_i, atom_j};
            Key down_up_key{-atv_ijk[rn][1], -atv_ijk[rn][2], -atv_ijk[rn][3], atom_j, atom_i};
            int orb_j = Total_NumOrbs[gh_AN];
            tasks.push_back({ct_AN, h_AN, atom_i, atom_j, orb_i, orb_j, rkey, down_up_key});
        }
    }

    int num_partials = 1;
#ifdef _OPENMP
    num_partials = std::max(1, omp_get_max_threads());
#endif
    std::vector<BlockMap> h_partials(static_cast<std::size_t>(num_partials));
    std::vector<BlockMap> s_partials(static_cast<std::size_t>(num_partials));
#ifdef _OPENMP
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for (std::size_t idx = 0; idx < tasks.size(); ++idx) {
            accumulate_openmx_neighbor_task(tasks[idx], SpinP_switch, h_partials[static_cast<std::size_t>(tid)], s_partials[static_cast<std::size_t>(tid)]);
        }
    }
#else
    for (std::size_t idx = 0; idx < tasks.size(); ++idx) {
        accumulate_openmx_neighbor_task(tasks[idx], SpinP_switch, h_partials[0], s_partials[0]);
    }
#endif
    for (auto& partial : h_partials) merge_block_maps(data.h_blocks, partial);
    for (auto& partial : s_partials) merge_block_maps(data.s_blocks, partial);
    if (timings != nullptr) (*timings)["build_openmx_blocks_s"] = seconds_since(t0);
    return data;
}
#endif

BlockMap hermitize_canonical(const BlockMap& blocks) {
    std::map<Key, bool> keys;
    for (const auto& [key, block] : blocks) {
        keys[key] = true;
        keys[mate_key(key)] = true;
    }
    BlockMap out;
    std::map<Key, bool> visited;
    for (const auto& [key, unused] : keys) {
        if (visited[key]) continue;
        Key mate = mate_key(key);
        visited[key] = true;
        visited[mate] = true;
        Key canon = canonical_key(key);
        Key canon_mate = mate_key(canon);
        auto block_it = blocks.find(canon);
        auto mate_it = blocks.find(canon_mate);
        if (canon == canon_mate) {
            if (block_it != blocks.end()) out[canon] = scale(add(block_it->second, dagger(block_it->second)), 0.5);
        } else if (block_it == blocks.end() && mate_it != blocks.end()) {
            out[canon] = scale(dagger(mate_it->second), 0.5);
        } else if (block_it != blocks.end() && mate_it == blocks.end()) {
            out[canon] = scale(block_it->second, 0.5);
        } else if (block_it != blocks.end() && mate_it != blocks.end()) {
            out[canon] = scale(add(block_it->second, dagger(mate_it->second)), 0.5);
        }
    }
    return out;
}

std::array<int, 3> transformed_rvec(const Operation& op, const Key& key) {
    std::array<double, 3> r{static_cast<double>(key.rx), static_cast<double>(key.ry), static_cast<double>(key.rz)};
    std::array<int, 3> out{};
    for (int a = 0; a < 3; ++a) {
        double value = 0.0;
        for (int b = 0; b < 3; ++b) value += op.rotation_frac[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] * r[static_cast<std::size_t>(b)];
        value += op.image_shifts[static_cast<std::size_t>(key.atom_j)][static_cast<std::size_t>(a)];
        value -= op.image_shifts[static_cast<std::size_t>(key.atom_i)][static_cast<std::size_t>(a)];
        double rounded = std::round(value);
        if (std::abs(value - rounded) > 1.0e-6) throw std::runtime_error("Symmetry operation produced non-integer R vector.");
        out[static_cast<std::size_t>(a)] = static_cast<int>(rounded);
    }
    return out;
}

BlockMap transform_canonical(const BlockMap& canonical, const SymmetryData& symmetry) {
    std::vector<std::pair<Key, Matrix>> items(canonical.begin(), canonical.end());
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif
    std::vector<BlockMap> partials(static_cast<std::size_t>(num_threads));
#ifdef _OPENMP
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(items.size()); ++idx) {
#else
    {
        int tid = 0;
        for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(items.size()); ++idx) {
#endif
            const Key& key = items[static_cast<std::size_t>(idx)].first;
            const Matrix& block = items[static_cast<std::size_t>(idx)].second;
            for (const auto& op : symmetry.operations) {
                auto rnew = transformed_rvec(op, key);
                Key target{rnew[0], rnew[1], rnew[2], op.atom_mappings[static_cast<std::size_t>(key.atom_i)], op.atom_mappings[static_cast<std::size_t>(key.atom_j)]};
                Key canon = canonical_key(target);
                Matrix new_block = transform_block(
                    op.orbital_blocks[static_cast<std::size_t>(key.atom_i)],
                    block,
                    op.orbital_blocks_dagger[static_cast<std::size_t>(key.atom_j)],
                    op.antiunitary
                );
                Matrix accum_block = (canon == target) ? new_block : dagger(new_block);
                auto [it, inserted] = partials[static_cast<std::size_t>(tid)].emplace(canon, Matrix(accum_block.rows, accum_block.cols));
                add_inplace(it->second, accum_block);
            }
        }
    }
    BlockMap out;
    for (const auto& partial : partials) {
        for (const auto& [key, block] : partial) {
            auto [it, inserted] = out.emplace(key, Matrix(block.rows, block.cols));
            add_inplace(it->second, block);
        }
    }
    return out;
}

BlockMap symmetrize_canonical(BlockMap canonical, const SymmetryData& symmetry) {
    BlockMap transformed = transform_canonical(canonical, symmetry);
    double factor = 1.0 / static_cast<double>(symmetry.operations.size());
    for (auto& [key, block] : transformed) block = scale(block, factor);
    return transformed;
}

BlockMap expand_hermitian(const BlockMap& canonical) {
    BlockMap out;
    for (const auto& [key, block] : canonical) {
        out[key] = block;
        Key mate = mate_key(key);
        if (!(mate == key)) out[mate] = dagger(block);
    }
    return out;
}

BlockMap prune_blocks(const BlockMap& blocks, double cutoff) {
    BlockMap out;
    for (const auto& [key, block] : blocks) {
        bool keep = false;
        for (const auto& value : block.data) {
            if (std::abs(value) > cutoff) {
                keep = true;
                break;
            }
        }
        if (keep) out[key] = block;
    }
    return out;
}

BlockMap symmetrize_blocks(const BlockMap& blocks, const SymmetryData& symmetry, double cutoff) {
    BlockMap canonical = hermitize_canonical(blocks);
    canonical = symmetrize_canonical(std::move(canonical), symmetry);
    canonical = symmetrize_canonical(std::move(canonical), symmetry);
    return prune_blocks(expand_hermitian(canonical), cutoff);
}

int global_index(int atom, int local, const MatrixSet& data) {
    if (!data.spinful) return data.atom_offsets[static_cast<std::size_t>(atom)] + local;
    int norb = data.atom_orbital_counts[static_cast<std::size_t>(atom)];
    int spin = local / norb;
    int local_orb = local % norb;
    return spin * data.nwann_spinless + data.atom_offsets[static_cast<std::size_t>(atom)] + local_orb;
}

int count_nonzero(const BlockMap& blocks, double cutoff) {
    int count = 0;
    for (const auto& [key, block] : blocks) {
        for (const auto& value : block.data) {
            if (std::abs(value) > cutoff) ++count;
        }
    }
    return count;
}

void write_sparse_dat(const std::filesystem::path& path, const std::string& label, const MatrixSet& data, const BlockMap& blocks, double cutoff) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Could not open output file: " + path.string());
    std::map<std::array<int, 3>, bool> rvecs;
    for (const auto& [key, block] : blocks) rvecs[{key.rx, key.ry, key.rz}] = true;
    out << " ! Sparse format of " << label << "\n";
    out << " " << count_nonzero(blocks, cutoff) << " ! Number of non-zeros lines of " << label << "mnR\n";
    out << " " << data.nwann << " ! Number of orbitals\n";
    out << " " << rvecs.size() << " ! Number of R points\n";
    out << std::fixed << std::setprecision(8);
    for (const auto& [key, block] : blocks) {
        for (int row = 0; row < block.rows; ++row) {
            for (int col = 0; col < block.cols; ++col) {
                auto value = block.at(row, col);
                if (std::abs(value) <= cutoff) continue;
                out << std::setw(5) << key.rx << std::setw(5) << key.ry << std::setw(5) << key.rz
                    << std::setw(6) << (global_index(key.atom_i, row, data) + 1)
                    << std::setw(6) << (global_index(key.atom_j, col, data) + 1)
                    << std::setw(16) << value.real() << std::setw(16) << value.imag() << "\n";
            }
        }
    }
}

void write_u16(std::ostream& out, std::uint16_t value) {
    char bytes[2] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
    };
    out.write(bytes, 2);
}

void write_u32(std::ostream& out, std::uint32_t value) {
    char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    out.write(bytes, 4);
}

void write_u64(std::ostream& out, std::uint64_t value) {
    char bytes[8] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
        static_cast<char>((value >> 32) & 0xffu),
        static_cast<char>((value >> 40) & 0xffu),
        static_cast<char>((value >> 48) & 0xffu),
        static_cast<char>((value >> 56) & 0xffu),
    };
    out.write(bytes, 8);
}

std::uint32_t crc32_update(std::uint32_t crc, const unsigned char* data, std::size_t size) {
    static std::uint32_t table[256];
    static bool initialized = false;
    if (!initialized) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int bit = 0; bit < 8; ++bit) {
                c = (c & 1u) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        initialized = true;
    }
    crc = crc ^ 0xffffffffu;
    for (std::size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
    }
    return crc ^ 0xffffffffu;
}

std::string npy_header(const std::string& descr, std::size_t count) {
    std::ostringstream shape;
    shape << count << ",";
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (" + shape.str() + "), }";
    constexpr std::size_t prefix = 10;  // magic + version + uint16 header length
    std::size_t padding = 16 - ((prefix + dict.size() + 1) % 16);
    if (padding == 16) padding = 0;
    dict.append(padding, ' ');
    dict.push_back('\n');
    if (dict.size() > 65535) throw std::runtime_error("NPY header too large for v1.0 format.");
    std::string header;
    header.append("\x93NUMPY", 6);
    header.push_back(static_cast<char>(1));
    header.push_back(static_cast<char>(0));
    header.push_back(static_cast<char>(dict.size() & 0xffu));
    header.push_back(static_cast<char>((dict.size() >> 8) & 0xffu));
    header += dict;
    return header;
}

std::uint32_t zip32_or_sentinel(std::uint64_t value) {
    return value > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(value);
}

template <typename T>
void write_npz_member(
    std::ofstream& out,
    std::vector<NpzEntry>& entries,
    const std::string& key,
    const std::string& descr,
    const std::vector<T>& values
) {
    std::string filename = key + ".npy";
    std::string header = npy_header(descr, values.size());
    const unsigned char* header_bytes = reinterpret_cast<const unsigned char*>(header.data());
    std::uint32_t crc = crc32_update(0, header_bytes, header.size());
    if (!values.empty()) {
        crc = crc32_update(
            crc,
            reinterpret_cast<const unsigned char*>(values.data()),
            values.size() * sizeof(T)
        );
    }
    std::uint64_t payload_size = header.size() + values.size() * sizeof(T);
    std::uint64_t offset = static_cast<std::uint64_t>(out.tellp());

    write_u32(out, 0x04034b50u);
    write_u16(out, 45);
    write_u16(out, 0);
    write_u16(out, 0);
    write_u16(out, 0);
    write_u16(out, 0);
    write_u32(out, crc);
    write_u32(out, 0xffffffffu);
    write_u32(out, 0xffffffffu);
    write_u16(out, static_cast<std::uint16_t>(filename.size()));
    write_u16(out, 20);
    out.write(filename.data(), static_cast<std::streamsize>(filename.size()));
    write_u16(out, 0x0001);
    write_u16(out, 16);
    write_u64(out, payload_size);
    write_u64(out, payload_size);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!values.empty()) {
        out.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T))
        );
    }
    entries.push_back({filename, crc, payload_size, payload_size, offset});
}

std::string rvec_key(const std::array<int, 3>& rvec) {
    return "(" + std::to_string(rvec[0]) + ", " + std::to_string(rvec[1]) + ", " + std::to_string(rvec[2]) + ")";
}

struct SparseTriplets {
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> cols;
    std::vector<std::complex<double>> vals;
};

std::map<std::array<int, 3>, SparseTriplets> collect_sparse_triplets(const MatrixSet& data, const BlockMap& blocks, double cutoff) {
    std::map<std::array<int, 3>, SparseTriplets> grouped;
    for (const auto& [key, block] : blocks) {
        auto& triplets = grouped[{key.rx, key.ry, key.rz}];
        for (int row = 0; row < block.rows; ++row) {
            for (int col = 0; col < block.cols; ++col) {
                auto value = block.at(row, col);
                if (std::abs(value) <= cutoff) continue;
                triplets.rows.push_back(static_cast<std::int32_t>(global_index(key.atom_i, row, data)));
                triplets.cols.push_back(static_cast<std::int32_t>(global_index(key.atom_j, col, data)));
                triplets.vals.push_back(value);
            }
        }
    }
    return grouped;
}

void write_sparse_npz(const std::filesystem::path& path, const MatrixSet& data, const BlockMap& blocks, double cutoff) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Could not open output file: " + path.string());
    std::vector<NpzEntry> entries;
    auto grouped = collect_sparse_triplets(data, blocks, cutoff);
    for (const auto& [rvec, triplets] : grouped) {
        std::string key = rvec_key(rvec);
        write_npz_member(out, entries, key + "_row", "<i4", triplets.rows);
        write_npz_member(out, entries, key + "_col", "<i4", triplets.cols);
        write_npz_member(out, entries, key + "_val", "<c16", triplets.vals);
    }

    std::uint64_t central_offset = static_cast<std::uint64_t>(out.tellp());
    for (const auto& entry : entries) {
        write_u32(out, 0x02014b50u);
        write_u16(out, 45);
        write_u16(out, 45);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u32(out, entry.crc);
        write_u32(out, 0xffffffffu);
        write_u32(out, 0xffffffffu);
        write_u16(out, static_cast<std::uint16_t>(entry.name.size()));
        write_u16(out, 28);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u32(out, 0);
        write_u32(out, 0xffffffffu);
        out.write(entry.name.data(), static_cast<std::streamsize>(entry.name.size()));
        write_u16(out, 0x0001);
        write_u16(out, 24);
        write_u64(out, entry.uncompressed_size);
        write_u64(out, entry.compressed_size);
        write_u64(out, entry.local_header_offset);
    }
    std::uint64_t central_size = static_cast<std::uint64_t>(out.tellp()) - central_offset;
    std::uint64_t zip64_eocd_offset = static_cast<std::uint64_t>(out.tellp());
    write_u32(out, 0x06064b50u);
    write_u64(out, 44);
    write_u16(out, 45);
    write_u16(out, 45);
    write_u32(out, 0);
    write_u32(out, 0);
    write_u64(out, entries.size());
    write_u64(out, entries.size());
    write_u64(out, central_size);
    write_u64(out, central_offset);
    write_u32(out, 0x07064b50u);
    write_u32(out, 0);
    write_u64(out, zip64_eocd_offset);
    write_u32(out, 1);
    write_u32(out, 0x06054b50u);
    write_u16(out, 0);
    write_u16(out, 0);
    write_u16(out, entries.size() > 65535 ? 0xffffu : static_cast<std::uint16_t>(entries.size()));
    write_u16(out, entries.size() > 65535 ? 0xffffu : static_cast<std::uint16_t>(entries.size()));
    write_u32(out, zip32_or_sentinel(central_size));
    write_u32(out, zip32_or_sentinel(central_offset));
    write_u16(out, 0);
}

double norm_blocks(const BlockMap& blocks) {
    long double total = 0.0;
    for (const auto& [key, block] : blocks) {
        for (const auto& value : block.data) total += std::norm(value);
    }
    return std::sqrt(static_cast<double>(total));
}

double hermiticity_residual(const BlockMap& blocks) {
    long double total = 0.0;
    for (const auto& [key, block] : blocks) {
        auto mate_it = blocks.find(mate_key(key));
        Matrix mate = mate_it == blocks.end() ? Matrix(block.cols, block.rows) : dagger(mate_it->second);
        Matrix diff = add(block, scale(mate, -1.0));
        for (const auto& value : diff.data) total += std::norm(value);
    }
    double denom = norm_blocks(blocks);
    if (denom == 0.0) return total == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    return std::sqrt(static_cast<double>(total)) / denom;
}

void write_report(
    const std::filesystem::path& path,
    const std::string& diagnostics,
    const std::string& output_format,
    const BlockMap& h_blocks,
    const BlockMap& s_blocks,
    double cutoff,
    const std::map<std::string, double>& timings
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Could not open report file: " + path.string());
    out << "{\n";
    out << "  \"diagnostics\": \"" << diagnostics << "\",\n";
    out << "  \"output_format\": \"" << output_format << "\",\n";
    out << "  \"cutoff\": " << std::setprecision(16) << cutoff << ",\n";
    out << "  \"H\": {\n";
    out << "    \"hermiticity_residual_after\": " << std::setprecision(16) << hermiticity_residual(h_blocks) << ",\n";
    out << "    \"output_nonzero\": " << count_nonzero(h_blocks, cutoff) << "\n";
    out << "  },\n";
    out << "  \"S\": {\n";
    out << "    \"hermiticity_residual_after\": " << std::setprecision(16) << hermiticity_residual(s_blocks) << ",\n";
    out << "    \"output_nonzero\": " << count_nonzero(s_blocks, cutoff) << "\n";
    out << "  },\n";
    out << "  \"timing_s\": {\n";
    bool first = true;
    for (const auto& [key, value] : timings) {
        if (!first) out << ",\n";
        first = false;
        out << "    \"" << key << "\": " << std::setprecision(9) << value;
    }
    out << "\n";
    out << "  }\n";
    out << "}\n";
}

struct Args {
    std::filesystem::path mock_input;
    std::filesystem::path symmetry_json;
    std::filesystem::path output_dir = ".";
    std::string diagnostics = "basic";
    std::string output_format = "dat";
    double cutoff = 1.0e-7;
    int threads = 0;
    std::string scfout;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };
        if (arg == "--mock-input") args.mock_input = require_value(arg);
        else if (arg == "--symmetry-json") args.symmetry_json = require_value(arg);
        else if (arg == "--output-dir") args.output_dir = require_value(arg);
        else if (arg == "--cutoff") args.cutoff = std::stod(require_value(arg));
        else if (arg == "--diagnostics") args.diagnostics = require_value(arg);
        else if (arg == "--output-format") args.output_format = require_value(arg);
        else if (arg == "--threads") args.threads = std::stoi(require_value(arg));
        else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: analysis_symm_hs [case.scfout] --symmetry-json FILE --output-dir DIR [--mock-input FILE] [--cutoff VALUE] [--diagnostics basic|full] [--output-format dat|npz|both] [--threads N]\n";
            std::exit(0);
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("Unknown option: " + arg);
        } else {
            args.scfout = arg;
        }
    }
    if (args.symmetry_json.empty()) throw std::runtime_error("--symmetry-json is required.");
    if (args.diagnostics != "basic" && args.diagnostics != "full") throw std::runtime_error("--diagnostics must be basic or full.");
    if (args.output_format != "dat" && args.output_format != "npz" && args.output_format != "both") {
        throw std::runtime_error("--output-format must be dat, npz, or both.");
    }
    return args;
}

void validate_compatible(const MatrixSet& data, const SymmetryData& symmetry) {
    if (data.spinful != symmetry.spinful) throw std::runtime_error("Spinful flag mismatch between input and symmetry JSON.");
    if (data.atom_orbital_counts != symmetry.atom_orbital_counts) throw std::runtime_error("Atom orbital counts mismatch between input and symmetry JSON.");
    if (data.nwann != symmetry.nwann) throw std::runtime_error("nwann mismatch between input and symmetry JSON.");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto total_start = Clock::now();
        std::map<std::string, double> timings;
        Args args = parse_args(argc, argv);
#ifdef _OPENMP
        if (args.threads > 0) omp_set_num_threads(args.threads);
#else
        if (args.threads > 0) std::cerr << "warning: --threads ignored because this binary was built without OpenMP\n";
#endif
        constexpr int kTotalSteps = 8;
        auto t0 = Clock::now();
        log_step_start(1, kTotalSteps, "load symmetry JSON");
        SymmetryData symmetry = load_symmetry(args.symmetry_json);
        timings["load_symmetry_json_s"] = seconds_since(t0);
        log_step_done(1, kTotalSteps, "load symmetry JSON", timings["load_symmetry_json_s"]);
        MatrixSet data;
        t0 = Clock::now();
        log_step_start(2, kTotalSteps, args.mock_input.empty() ? "read scfout and build blocks" : "read mock input and build blocks");
        if (!args.mock_input.empty()) {
            data = load_mock_input(args.mock_input);
        } else {
#ifdef OPENMX_ANALYSIS_BUILD
            ScopedStdoutSilencer silence_openmx_banner(true);
            data = load_openmx_scfout(argc, argv, &timings);
#else
            throw std::runtime_error("This binary was not built with OPENMX_ANALYSIS_BUILD; use --mock-input in standalone tests.");
#endif
        }
        timings["read_input_build_blocks_s"] = seconds_since(t0);
        log_step_done(2, kTotalSteps, args.mock_input.empty() ? "read scfout and build blocks" : "read mock input and build blocks", timings["read_input_build_blocks_s"]);
        t0 = Clock::now();
        log_step_start(3, kTotalSteps, "validate input against symmetry");
        validate_compatible(data, symmetry);
        timings["validate_s"] = seconds_since(t0);
        log_step_done(3, kTotalSteps, "validate input against symmetry", timings["validate_s"]);
        std::filesystem::create_directories(args.output_dir);
        t0 = Clock::now();
        log_step_start(4, kTotalSteps, "symmetrize H");
        BlockMap h_sym = symmetrize_blocks(data.h_blocks, symmetry, args.cutoff);
        timings["symmetrize_H_s"] = seconds_since(t0);
        log_step_done(4, kTotalSteps, "symmetrize H", timings["symmetrize_H_s"]);
        t0 = Clock::now();
        log_step_start(5, kTotalSteps, "symmetrize S");
        BlockMap s_sym = symmetrize_blocks(data.s_blocks, symmetry, args.cutoff);
        timings["symmetrize_S_s"] = seconds_since(t0);
        log_step_done(5, kTotalSteps, "symmetrize S", timings["symmetrize_S_s"]);
        std::vector<std::filesystem::path> written_paths;
        if (args.output_format == "dat" || args.output_format == "both") {
            t0 = Clock::now();
            log_step_start(6, kTotalSteps, "write DAT outputs");
            auto h_path = args.output_dir / "H.dat";
            write_sparse_dat(h_path, "H", data, h_sym, args.cutoff);
            timings["write_H_dat_s"] = seconds_since(t0);
            written_paths.push_back(h_path);
            t0 = Clock::now();
            auto s_path = args.output_dir / "S.dat";
            write_sparse_dat(s_path, "S", data, s_sym, args.cutoff);
            timings["write_S_dat_s"] = seconds_since(t0);
            written_paths.push_back(s_path);
            log_step_done(6, kTotalSteps, "write DAT outputs", timings["write_H_dat_s"] + timings["write_S_dat_s"]);
        } else {
            timings["write_H_dat_s"] = 0.0;
            timings["write_S_dat_s"] = 0.0;
            log_step_skip(6, kTotalSteps, "write DAT outputs");
        }
        if (args.output_format == "npz" || args.output_format == "both") {
            t0 = Clock::now();
            log_step_start(7, kTotalSteps, "write NPZ outputs");
            auto h_path = args.output_dir / "H.npz";
            write_sparse_npz(h_path, data, h_sym, args.cutoff);
            timings["write_H_npz_s"] = seconds_since(t0);
            written_paths.push_back(h_path);
            t0 = Clock::now();
            auto s_path = args.output_dir / "S.npz";
            write_sparse_npz(s_path, data, s_sym, args.cutoff);
            timings["write_S_npz_s"] = seconds_since(t0);
            written_paths.push_back(s_path);
            log_step_done(7, kTotalSteps, "write NPZ outputs", timings["write_H_npz_s"] + timings["write_S_npz_s"]);
        } else {
            timings["write_H_npz_s"] = 0.0;
            timings["write_S_npz_s"] = 0.0;
            log_step_skip(7, kTotalSteps, "write NPZ outputs");
        }
        t0 = Clock::now();
        log_step_start(8, kTotalSteps, "write report");
        timings["total_wall_before_report_s"] = seconds_since(total_start);
        timings["total_wall_s"] = timings["total_wall_before_report_s"];
        auto report_path = args.output_dir / "symmetrization_report.json";
        write_report(report_path, args.diagnostics, args.output_format, h_sym, s_sym, args.cutoff, timings);
        timings["write_report_s"] = seconds_since(t0);
        timings["total_wall_s"] = seconds_since(total_start);
        log_step_done(8, kTotalSteps, "write report", timings["write_report_s"]);
        for (const auto& path : written_paths) {
            std::cout << "Wrote " << path << "\n";
        }
        std::cout << "Wrote " << report_path << "\n";
        std::cout << "analysis_symm_hs finished in " << format_seconds(timings["total_wall_s"]) << " s" << std::endl;
    } catch (const std::exception& exc) {
        std::cerr << "analysis_symm_hs: error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
