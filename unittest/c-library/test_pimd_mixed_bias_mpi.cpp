// Mixed-bead force pullback and package-independent bias lifecycle.
#define LAMMPS_LIB_MPI 1
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "lammps.h"
#include "library.h"
#include "modify.h"
#include "pimd_bias_interface.h"
#include "universe.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

using namespace LAMMPS_NS;
namespace {
enum class Behavior { NORMAL, OMIT, DUPLICATE, STALE, COUNT, NONFINITE, VIRIAL, ENERGY };
class MixedBias : public Fix {
public:
    std::shared_ptr<PIMDBiasRegistration> token;
    PIMDBiasMode mode        = PIMDBiasMode::BIAS_MEAN;
    Behavior behavior        = Behavior::NORMAL;
    std::uint64_t generation = 0;
    double energy            = 0.0;
    bool contraction         = false;

    MixedBias(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
    {
        scalar_flag = extscalar = energy_global_flag = virial_global_flag = 1;
        thermo_energy = thermo_virial = global_freq = 1;
    }
    ~MixedBias() override
    {
        if (token && token->owner) token->owner->unregister_bias(*token);
    }
    int setmask() override { return FixConst::POST_FORCE; }
    void init() override
    {
        int dim;
        auto *core = modify->get_fix_by_id("pi");
        auto *api  = static_cast<PIMDBiasInterface *>(core->extract("pimd_bias_interface", dim));
        token      = api->register_bias(this, mode, contraction);
    }
    void setup(int vflag) override { post_force(vflag); }
    double compute_scalar() override { return energy; }
    void post_force(int /*vflag*/) override
    {
        auto *api       = token->owner;
        const auto view = api->bias_coordinates(*token);
        generation      = view.generation;
        if (behavior == Behavior::OMIT) return;
        const int p = api->bias_capabilities().beads;
        std::array<double, 9> x{};
        for (int i = 0; i < view.nlocal; ++i)
            for (int d = 0; d < 3; ++d)
                x[3 * (view.tags[i] - 1) + d] = view.beads[i][d];
        MPI_Allreduce(MPI_IN_PLACE, x.data(), 9, MPI_DOUBLE, MPI_SUM, world);
        if (mode == PIMDBiasMode::COORDINATE_MEAN)
            for (int i = 0; i < 9; ++i)
                x[i] = view.centroid_by_tag[i];
        double s = 0.0;
        for (int d = 0; d < 3; ++d)
            for (int j = 0; j < 2; ++j)
                s += std::pow(x[3 * j + d] - x[3 * (j + 1) + d], 2);
        double mean = s / (p * comm->nprocs);
        MPI_Allreduce(MPI_IN_PLACE, &mean, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
        const double cv = mode == PIMDBiasMode::CV_MEAN ? mean : s;
        energy          = 0.01 * cv * cv;
        if (mode == PIMDBiasMode::BIAS_MEAN)
            energy /= p;
        else if (universe->iworld != 0)
            energy = 0.0;
        std::array<double, 9> force{};
        for (int d = 0; d < 3; ++d)
            for (int j = 0; j < 2; ++j) {
                const double g = -0.04 * cv * (x[3 * j + d] - x[3 * (j + 1) + d]) / p;
                force[3 * j + d] += g;
                force[3 * (j + 1) + d] -= g;
            }
        std::vector<double> local(std::max(1, 3 * view.nlocal), 0.0);
        std::array<double, 6> v{};
        for (int i = 0; i < view.nlocal; ++i) {
            const int j = 3 * (view.tags[i] - 1);
            for (int d = 0; d < 3; ++d)
                local[3 * i + d] = force[j + d];
            v[0] += x[j] * force[j];
            v[1] += x[j + 1] * force[j + 1];
            v[2] += x[j + 2] * force[j + 2];
            v[3] += x[j + 1] * force[j];
            v[4] += x[j + 2] * force[j];
            v[5] += x[j + 2] * force[j + 1];
        }
        PIMDBiasSubmission submission{generation, view.nlocal, local.data(), energy, v.data()};
        if (universe->me == 1) {
            if (behavior == Behavior::STALE) --submission.generation;
            if (behavior == Behavior::COUNT) ++submission.nlocal;
            if (behavior == Behavior::VIRIAL) v[0] = std::numeric_limits<double>::infinity();
            if (behavior == Behavior::ENERGY)
                submission.energy = std::numeric_limits<double>::quiet_NaN();
        }
        if (behavior == Behavior::NONFINITE && view.nlocal)
            local[0] = std::numeric_limits<double>::quiet_NaN();
        api->submit_bias(*token, submission, virial);
        if (behavior == Behavior::DUPLICATE) api->submit_bias(*token, submission, virial);
    }
};
Fix *make_bias(LAMMPS *lmp, int narg, char **arg)
{
    return new MixedBias(lmp, narg, arg);
}

class MixedBiasMPI : public ::testing::Test {
protected:
    LAMMPS *lmp = nullptr;
    int rank    = 0;
    void SetUp() override
    {
        int size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        ASSERT_EQ(size, 8);
        Modify::fix_styles().set_plugin("MIXED/TEST/BIAS", make_bias);
    }
    void TearDown() override
    {
        if (lmp) lammps_close(lmp);
        Modify::fix_styles().clear_plugin("MIXED/TEST/BIAS");
    }
    bool command(const std::string &text)
    {
        lammps_commands_string(lmp, text.c_str());
        int bad = lammps_has_error(lmp);
        char message[1024]{};
        if (bad) lammps_get_last_error_message(lmp, message, sizeof(message));
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        EXPECT_EQ(bad, 0) << message;
        return !bad;
    }
    void open()
    {
        LAMMPS::argv args = {"mixed_bias", "-screen", "none", "-log", "none",
                             "-partition", "4x2",     "-in",  "none", "-nocite"};
        lmp               = new LAMMPS(args, MPI_COMM_WORLD);
        lammps_set_show_error(lmp, 0);
    }
    bool prepare(int coarse = 2, bool pile = false)
    {
        open();
        std::ostringstream in;
        const int b = rank / 2;
        in << std::setprecision(17)
           << "units lj\natom_style atomic\natom_modify map array sort 0 0\nprocessors 2 1 1\n"
           << "region box block 0 12 0 12 0 12\ncreate_box 3 box\n"
           << "create_atoms 1 single " << 5.7 + 0.2 * b << " 4.1 4.3\n"
           << "create_atoms 2 single " << 4.8 + 0.3 * (b / (4 / coarse)) << " 4.4 4.5\n"
           << "create_atoms 3 single 4.0 4.7 4.2\n"
           << "mass 1 1\nmass 2 2\nmass 3 3\npair_style zero 2.5\npair_coeff * *\n"
           << "group fine id 1\ngroup coarse id 2\nvelocity all set 0.01 0.02 -0.01\n"
           << "timestep 0.0001\nthermo 100000\nfix pi all pimd/mixed temp 1 sp 6.283185307179586 "
           << "beads fine 4 beads coarse " << coarse;
        if (pile) in << " thermostat pile 1 19771";
        return command(in.str());
    }
    MixedBias *add(PIMDBiasMode mode = PIMDBiasMode::BIAS_MEAN)
    {
        command("fix b all MIXED/TEST/BIAS");
        auto *fix = static_cast<MixedBias *>(lmp->modify->get_fix_by_id("b"));
        fix->mode = mode;
        return fix;
    }
    void error(const std::string &text, const std::string &part)
    {
        lammps_commands_string(lmp, text.c_str());
        EXPECT_EQ(lammps_has_error(lmp), 1);
        char message[1024]{};
        lammps_get_last_error_message(lmp, message, sizeof(message));
        EXPECT_NE(std::string(message).find(part), std::string::npos) << message;
    }
    std::array<double, 108> state()
    {
        std::array<double, 108> data{};
        for (int i = 0; i < lmp->atom->nlocal; ++i) {
            const int j = 27 * (rank / 2) + 9 * (lmp->atom->tag[i] - 1);
            for (int d = 0; d < 3; ++d) {
                data[j + d]     = lmp->atom->x[i][d];
                data[j + 3 + d] = lmp->atom->v[i][d];
                data[j + 6 + d] = lmp->atom->f[i][d];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, data.data(), data.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return data;
    }
    static double analytic_force(std::array<double, 108> q, PIMDBiasMode mode, int atom, int bead,
                                 int dim, int count)
    {
        if (mode == PIMDBiasMode::COORDINATE_MEAN)
            for (int i = 0; i < 3; ++i)
                for (int d = 0; d < 3; ++d) {
                    double mean = 0.0;
                    for (int b = 0; b < 4; ++b)
                        mean += q[27 * b + 9 * i + d] / 4;
                    for (int b = 0; b < 4; ++b)
                        q[27 * b + 9 * i + d] = mean;
                }
        std::array<double, 4> cv{};
        double mean = 0.0;
        for (int b = 0; b < 4; ++b) {
            for (int d = 0; d < 3; ++d)
                cv[b] += std::pow(q[27 * b + d] - q[27 * b + 9 + d], 2) +
                         std::pow(q[27 * b + 9 + d] - q[27 * b + 18 + d], 2);
            mean += cv[b] / 4;
        }
        double derivative = 0.0;
        for (int b = bead * 4 / count; b < (bead + 1) * 4 / count; ++b) {
            const double x = q[27 * b + dim], y = q[27 * b + 9 + dim], z = q[27 * b + 18 + dim];
            const double jacobian = atom == 0 ? 2 * (x - y)
                                              : (atom == 1 ? 2 * (2 * y - x - z) : 2 * (z - y));
            derivative += 0.02 * (mode == PIMDBiasMode::CV_MEAN ? mean : cv[b]) * jacobian / 4;
        }
        return -count * derivative;
    }
    static double oracle(std::array<double, 108> q, PIMDBiasMode mode)
    {
        if (mode == PIMDBiasMode::COORDINATE_MEAN)
            for (int i = 0; i < 3; ++i)
                for (int d = 0; d < 3; ++d) {
                    double mean = 0.0;
                    for (int b = 0; b < 4; ++b)
                        mean += q[27 * b + 9 * i + d] / 4;
                    for (int b = 0; b < 4; ++b)
                        q[27 * b + 9 * i + d] = mean;
                }
        double mean_s = 0.0, mean_s2 = 0.0;
        for (int b = 0; b < 4; ++b) {
            double s = 0.0;
            for (int d = 0; d < 3; ++d)
                s += std::pow(q[27 * b + d] - q[27 * b + 9 + d], 2) +
                     std::pow(q[27 * b + 9 + d] - q[27 * b + 18 + d], 2);
            mean_s += s / 4;
            mean_s2 += s * s / 4;
        }
        return 0.01 * (mode == PIMDBiasMode::CV_MEAN ? mean_s * mean_s : mean_s2);
    }
};

TEST_F(MixedBiasMPI, unique_coordinate_gradient_energy_and_estimators)
{
    for (int coarse : {1, 2})
        for (auto mode :
             {PIMDBiasMode::COORDINATE_MEAN, PIMDBiasMode::CV_MEAN, PIMDBiasMode::BIAS_MEAN})
            for (bool tally : {false, true}) {
                SCOPED_TRACE(std::to_string(coarse) + ":" + std::to_string(int(mode)) + ":" +
                             std::to_string(tally));
                ASSERT_TRUE(prepare(coarse));
                add(mode);
                ASSERT_TRUE(command(tally ? "fix_modify b energy yes\nrun 0"
                                          : "fix_modify b energy no\nrun 0"));
                auto q            = state();
                auto *pi          = lmp->modify->get_fix_by_id("pi");
                const double bias = oracle(q, mode);
                EXPECT_NEAR(pi->compute_vector(0), 0, 1e-11);
                EXPECT_NEAR(pi->compute_vector(6), bias, 1e-11);
                EXPECT_NEAR(pi->compute_vector(5), 4.5, 1e-11);
                EXPECT_NEAR(pi->compute_scalar(),
                            pi->compute_vector(1) + pi->compute_vector(2) + bias, 1e-11);
                for (int i = 0; i < 3; ++i) {
                    const int p = i == 0 ? 4 : (i == 1 ? coarse : 1);
                    for (int a = 0; a < p; ++a)
                        for (int d = 0; d < 3; ++d) {
                            const double analytic = analytic_force(q, mode, i, a, d, p);
                            for (int b = a * 4 / p; b < (a + 1) * 4 / p; ++b)
                                EXPECT_NEAR(q[27 * b + 9 * i + 6 + d], analytic, 1e-11);
                            for (double delta : {1e-4, 1e-5, 1e-6}) {
                                auto plus = q, minus = q;
                                for (int b = a * 4 / p; b < (a + 1) * 4 / p; ++b) {
                                    plus[27 * b + 9 * i + d] += delta;
                                    minus[27 * b + 9 * i + d] -= delta;
                                }
                                const double expected =
                                    -p * (oracle(plus, mode) - oracle(minus, mode)) / (2 * delta);
                                for (int b = a * 4 / p; b < (a + 1) * 4 / p; ++b)
                                    EXPECT_NEAR(q[27 * b + 9 * i + 6 + d], expected, 5e-7);
                            }
                        }
                }
                EXPECT_GT(std::abs(q[24]), 1e-4); // Nonzero classical environment response.
                lammps_close(lmp);
                lmp = nullptr;
            }
}

TEST_F(MixedBiasMPI, repeated_step_unregister_and_growth)
{
    ASSERT_TRUE(prepare());
    auto *b = add();
    ASSERT_TRUE(command("run 0"));
    const auto g      = b->generation;
    const auto token  = b->token;
    const auto before = state();
    lmp->atom->avec->grow(lmp->atom->nmax + 64);
    ASSERT_TRUE(command("run 0"));
    EXPECT_GT(b->generation, g);
    auto after = state();
    for (int i = 0; i < 108; ++i)
        EXPECT_NEAR(before[i], after[i], 1e-12);
    ASSERT_TRUE(command("unfix b\nrun 0"));
    EXPECT_EQ(token->owner, nullptr);
    after = state();
    for (int bead = 0; bead < 4; ++bead)
        for (int i = 0; i < 3; ++i)
            for (int d = 0; d < 3; ++d)
                EXPECT_NEAR(after[27 * bead + 9 * i + 6 + d], 0, 1e-12);
    b = add();
    ASSERT_TRUE(command("run 0"));
    EXPECT_NE(b->token, token);
    EXPECT_GT(lmp->modify->get_fix_by_id("pi")->compute_vector(6), 0.0);
}

TEST_F(MixedBiasMPI, missing_duplicate_stale_and_invalid)
{
    for (auto behavior : {Behavior::OMIT, Behavior::DUPLICATE, Behavior::STALE, Behavior::COUNT,
                          Behavior::NONFINITE, Behavior::VIRIAL, Behavior::ENERGY}) {
        ASSERT_TRUE(prepare());
        auto *b     = add();
        b->behavior = behavior;
        if (rank == 1) EXPECT_EQ(lmp->atom->nlocal, 0);
        error("run 0", behavior == Behavior::OMIT        ? "Missing external"
                       : behavior == Behavior::DUPLICATE ? "Duplicate external"
                       : behavior == Behavior::STALE     ? "Stale external"
                                                         : "Invalid external");
        lammps_close(lmp);
        lmp = nullptr;
    }
}

TEST_F(MixedBiasMPI, unsupported_contraction_and_companion)
{
    ASSERT_TRUE(prepare());
    auto *b        = add();
    b->contraction = true;
    error("run 0", "contraction requires");
    b->contraction = false;
    ASSERT_TRUE(command("run 0"));
    ASSERT_TRUE(command("fix extra all nve"));
    error("run 0", "companion fixes");
}

TEST_F(MixedBiasMPI, core_deleted_before_adapter)
{
    ASSERT_TRUE(prepare());
    auto *b = add();
    ASSERT_TRUE(command("run 0"));
    const auto token = b->token;
    ASSERT_TRUE(command("unfix pi"));
    EXPECT_EQ(token->owner, nullptr);
    ASSERT_TRUE(command("unfix b"));
}

TEST_F(MixedBiasMPI, deterministic_and_pile_bias_restart)
{
    for (bool pile : {false, true}) {
        ASSERT_TRUE(prepare(2, pile));
        add(PIMDBiasMode::CV_MEAN);
        ASSERT_TRUE(command("run 20"));
        const auto full = state();
        lammps_close(lmp);
        lmp = nullptr;
        ASSERT_TRUE(prepare(2, pile));
        add(PIMDBiasMode::CV_MEAN);
        ASSERT_TRUE(command("run 10\nvariable file world mixed_bias.0 mixed_bias.1 mixed_bias.2 "
                            "mixed_bias.3\nwrite_restart ${file}"));
        lammps_close(lmp);
        lmp = nullptr;
        open();
        ASSERT_TRUE(command("processors 2 1 1\nvariable file world mixed_bias.0 mixed_bias.1 "
                            "mixed_bias.2 mixed_bias.3\nread_restart ${file}\natom_modify sort 0 "
                            "0\ntimestep 0.0001\nthermo 100000"));
        ASSERT_TRUE(command(
            std::string(
                "fix pi all pimd/mixed temp 1 sp 6.283185307179586 beads fine 4 beads coarse 2") +
            (pile ? " thermostat pile 1 19771" : "")));
        add(PIMDBiasMode::CV_MEAN);
        ASSERT_TRUE(command("run 10"));
        const auto split = state();
        for (int i = 0; i < 108; ++i)
            EXPECT_NEAR(full[i], split[i], 1e-10);
        lammps_close(lmp);
        lmp = nullptr;
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            for (int i = 0; i < 4; ++i)
                std::remove(("mixed_bias." + std::to_string(i)).c_str());
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
} // namespace
