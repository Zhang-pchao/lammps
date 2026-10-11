// Native lifecycle tests for the package-independent PIMD bias contract.

#define LAMMPS_LIB_MPI 1
#include "REPLICA/fix_pimd_langevin.h"
#include "atom.h"
#include "atom_vec.h"
#include "error.h"
#include "exceptions.h"
#include "fix.h"
#include "lammps.h"
#include "library.h"
#include "modify.h"
#include "pimd_bias_interface.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

namespace {

using namespace LAMMPS_NS;

enum class MockBehavior { SUBMIT, OMIT, DUPLICATE, STALE, STALE_ONE_RANK, WRONG_COUNT };

class FixTestPIMDBias : public Fix {
public:
    std::shared_ptr<PIMDBiasRegistration> registration;
    std::vector<std::uint64_t> generations;
    std::vector<bigint> timesteps;
    std::array<double, 3> increment{0.125, -0.25, 0.375};
    MockBehavior behavior    = MockBehavior::SUBMIT;
    bool harmonic_bias       = false;
    double harmonic_strength = 0.125;
    double physical_energy   = 0.0;
    int completed            = 0;

    struct Observation {
        tagint tag;
        std::array<double, 3> bead, unwrapped, centroid, force;
    };
    bool capture_coordinates = false;
    std::vector<Observation> observations;

    FixTestPIMDBias(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
    {
        if (narg != 4) error->all(FLERR, "Test PIMD bias requires a core fix ID");
        core_id            = arg[3];
        scalar_flag        = 1;
        extscalar          = 1;
        energy_global_flag = virial_global_flag = 1;
        thermo_energy = thermo_virial = 1;
        global_freq                   = 1;
    }

    ~FixTestPIMDBias() override
    {
        if (registration && registration->owner)
            registration->owner->unregister_bias(*registration);
    }

    int setmask() override { return FixConst::POST_FORCE; }

    void init() override
    {
        auto *core = modify->get_fix_by_id(core_id);
        if (!core) error->all(FLERR, "Test PIMD bias core fix is missing");
        int dim   = -1;
        auto *api = static_cast<PIMDBiasInterface *>(core->extract("pimd_bias_interface", dim));
        if (!api || dim != 0) error->all(FLERR, "Test PIMD bias core interface is missing");
        const auto mode = harmonic_bias ? PIMDBiasMode::BIAS_MEAN : PIMDBiasMode::COORDINATE_MEAN;
        registration    = api->register_bias(this, mode, false);
    }

    void setup(int vflag) override { post_force(vflag); }

    void post_force(int vflag) override
    {
        v_init(vflag);
        auto *api       = registration->owner;
        const auto view = api->bias_coordinates(*registration);
        generations.push_back(view.generation);
        timesteps.push_back(view.timestep);
        if (behavior == MockBehavior::OMIT) return;
        const int beads = api->bias_capabilities().beads;
        std::vector<double> contribution(3 * view.nlocal, 0.0);
        std::array<double, 6> physical_virial{};
        double local_energy = 0.0;
        if (capture_coordinates) observations.clear();
        // Two independent analytic potentials: a uniform centroid load, or
        // B = k/(2P) sum_b sum_i |x_bi|^2 with bead-dependent forces.
        for (int i = 0; i < view.nlocal; ++i) {
            const double *x = view.beads[i];
            for (int d = 0; d < 3; ++d) {
                const double force      = harmonic_bias ? -harmonic_strength * x[d] : increment[d];
                contribution[3 * i + d] = force / beads;
                local_energy += harmonic_bias ? 0.5 * harmonic_strength * x[d] * x[d] / beads
                                              : -x[d] * increment[d] / beads;
            }
            physical_virial[0] += x[0] * contribution[3 * i];
            physical_virial[1] += x[1] * contribution[3 * i + 1];
            physical_virial[2] += x[2] * contribution[3 * i + 2];
            physical_virial[3] += x[1] * contribution[3 * i];
            physical_virial[4] += x[2] * contribution[3 * i];
            physical_virial[5] += x[2] * contribution[3 * i + 1];
            if (capture_coordinates) {
                Observation observed{};
                observed.tag = view.tags[i];
                for (int d = 0; d < 3; ++d) {
                    observed.bead[d]      = view.beads[i][d];
                    observed.unwrapped[d] = view.unwrapped_beads[i][d];
                    observed.centroid[d]  = view.centroid_by_tag[3 * (view.tags[i] - 1) + d];
                    observed.force[d]     = contribution[3 * i + d];
                }
                observations.push_back(observed);
            }
        }
        MPI_Allreduce(&local_energy, &physical_energy, 1, MPI_DOUBLE, MPI_SUM, world);
        auto generation = view.generation;
        int universe_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &universe_rank);
        if (((behavior == MockBehavior::STALE) ||
             ((behavior == MockBehavior::STALE_ONE_RANK) && (universe_rank == 3))) &&
            generations.size() > 1)
            generation = generations[generations.size() - 2];
        PIMDBiasSubmission submission{generation, view.nlocal, contribution.data(), physical_energy,
                                      physical_virial.data()};
        if (behavior == MockBehavior::WRONG_COUNT) ++submission.nlocal;
        api->submit_bias(*registration, submission, virial);
        ++completed;
        if (behavior == MockBehavior::DUPLICATE)
            api->submit_bias(*registration, submission, virial);
    }

    double compute_scalar() override { return physical_energy; }

private:
    std::string core_id;
};

Fix *make_mock_bias(LAMMPS *lmp, int narg, char **arg)
{
    return new FixTestPIMDBias(lmp, narg, arg);
}

class FixTestPIMDCore : public FixPIMDLangevin {
public:
    int hook_calls              = 0;
    int final_calls             = 0;
    int hooks_before_final_kick = 0;
    std::vector<std::array<double, 3>> hook_force;

    FixTestPIMDCore(LAMMPS *lmp, int narg, char **arg) : FixPIMDLangevin(lmp, narg, arg) {}

    std::array<double, 6> current_stress()
    {
        compute_stress_tensor();
        std::array<double, 6> result{};
        for (int d = 0; d < 6; ++d)
            result[d] = stress_tensor[d];
        return result;
    }

    void final_integrate() override
    {
        ++final_calls;
        hooks_before_final_kick = hook_calls;
        FixPIMDLangevin::final_integrate();
    }

protected:
    void after_force_transform_hook() override
    {
        ++hook_calls;
        hook_force.resize(atom->nlocal);
        for (int i = 0; i < atom->nlocal; ++i)
            for (int d = 0; d < 3; ++d)
                hook_force[i][d] = atom->f[i][d];
    }
};

Fix *make_mock_core(LAMMPS *lmp, int narg, char **arg)
{
    return new FixTestPIMDCore(lmp, narg, arg);
}

class PIMDBiasMPI : public ::testing::Test {
protected:
    void *handle = nullptr;
    int rank     = 0;

    void SetUp() override
    {
        int nprocs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        ASSERT_EQ(nprocs, 4);
        Modify::fix_styles().set_plugin("PIMD/TEST/BIAS", &make_mock_bias);
        Modify::fix_styles().set_plugin("PIMD/TEST/CORE", &make_mock_core);
        open_partition();
        lammps_command(handle, "variable x2 world 15.0 6.0");
        lammps_command(handle, "variable y2 world 10.0 14.0");
        lammps_command(handle, "units lj");
        lammps_command(handle, "atom_style atomic");
        lammps_command(handle, "atom_modify map array");
        lammps_command(handle, "processors 2 1 1");
        lammps_command(handle, "region box block 0 20 0 20 0 20");
        lammps_command(handle, "create_box 1 box");
        lammps_command(handle, "create_atoms 1 single 5.0 10.0 2.0");
        lammps_command(handle, "create_atoms 1 single ${x2} ${y2} 2.0");
        lammps_command(handle, "mass 1 1.0");
        lammps_command(handle, "pair_style lj/cut 2.5");
        lammps_command(handle, "pair_coeff * * 0.0 1.0");
        lammps_command(handle, "velocity all set 0.0 0.0 0.0");
        lammps_command(handle, "timestep 0.0001");
        lammps_command(handle, "thermo_style custom step");
        lammps_command(handle, "thermo 0");
        lammps_set_show_error(handle, 0);
    }

    void TearDown() override
    {
        if (handle) lammps_close(handle);
        Modify::fix_styles().clear_plugin("PIMD/TEST/BIAS");
        Modify::fix_styles().clear_plugin("PIMD/TEST/CORE");
    }

    void open_partition()
    {
        const char *args[] = {"LAMMPS_test", "-screen", "none", "-log",    "none", "-partition",
                              "2x2",         "-in",     "none", "-nocite", nullptr};
        handle             = lammps_open(10, const_cast<char **>(args), MPI_COMM_WORLD, nullptr);
        ASSERT_NE(handle, nullptr);
    }

    void add_core(const char *method = "nmpimd", const char *style = "pimd/langevin")
    {
        std::string command = "fix fpimd all " + std::string(style) + " method " + method +
                              " ensemble nvt integrator obabo thermostat PILE_L 1234 tau 1.0e100 "
                              "temp 1.0 sp 1.0 fmass 1.0 fixcom no";
        if (std::string(method) == "nmpimd") command += " scale 0.0";
        lammps_command(handle, command.c_str());
    }

    std::array<double, 36> state()
    {
        std::array<double, 36> result{};
        auto *lmp        = static_cast<LAMMPS *>(handle);
        const int offset = 18 * (rank / 2);
        for (int i = 0; i < lmp->atom->nlocal; ++i) {
            const int atom_index = lmp->atom->tag[i] - 1;
            for (int d = 0; d < 3; ++d) {
                result[offset + 3 * atom_index + d]      = lmp->atom->x[i][d];
                result[offset + 6 + 3 * atom_index + d]  = lmp->atom->v[i][d];
                result[offset + 12 + 3 * atom_index + d] = lmp->atom->f[i][d];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, result.data(), static_cast<int>(result.size()), MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        return result;
    }

    void write_checkpoint(const std::string &prefix)
    {
        const std::string variable = "variable checkpoint world " + prefix + ".0 " + prefix + ".1";
        lammps_command(handle, variable.c_str());
        lammps_command(handle, "write_restart ${checkpoint}");
        lammps_command(handle, "variable checkpoint delete");
        EXPECT_EQ(lammps_has_error(handle), 0);
    }

    void read_checkpoint(const std::string &prefix)
    {
        lammps_close(handle);
        handle = nullptr;
        open_partition();
        const std::string variable = "variable checkpoint world " + prefix + ".0 " + prefix + ".1";
        lammps_command(handle, variable.c_str());
        lammps_command(handle, "read_restart ${checkpoint}");
        lammps_command(handle, "variable checkpoint delete");
        lammps_command(handle, "thermo_style custom step");
        lammps_command(handle, "thermo 0");
        lammps_set_show_error(handle, 0);
        add_core();
        add_bias();
        EXPECT_EQ(lammps_has_error(handle), 0);
    }

    FixTestPIMDBias *add_bias(const char *id = "bias")
    {
        const std::string command = "fix " + std::string(id) + " all PIMD/TEST/BIAS fpimd";
        lammps_command(handle, command.c_str());
        return static_cast<FixTestPIMDBias *>(
            static_cast<LAMMPS *>(handle)->modify->get_fix_by_id(id));
    }

    PIMDBiasInterface *interface()
    {
        auto *core = static_cast<LAMMPS *>(handle)->modify->get_fix_by_id("fpimd");
        int dim    = -1;
        auto *api  = static_cast<PIMDBiasInterface *>(core->extract("pimd_bias_interface", dim));
        EXPECT_EQ(dim, 0);
        return api;
    }

    void expect_command_error(const char *command, const char *expected)
    {
        lammps_command(handle, command);
        EXPECT_EQ(lammps_has_error(handle), 1);
        char message[1024]{};
        EXPECT_NE(lammps_get_last_error_message(handle, message, sizeof(message)), 0);
        EXPECT_NE(std::string(message).find(expected), std::string::npos);
    }

    std::array<double, 6> forces()
    {
        std::array<double, 6> result{};
        auto *lmp = static_cast<LAMMPS *>(handle);
        for (int i = 0; i < lmp->atom->nlocal; ++i)
            for (int d = 0; d < 3; ++d)
                result[3 * (lmp->atom->tag[i] - 1) + d] = lmp->atom->f[i][d];
        MPI_Allreduce(MPI_IN_PLACE, result.data(), static_cast<int>(result.size()), MPI_DOUBLE,
                      MPI_SUM, lmp->world);
        return result;
    }
};

TEST_F(PIMDBiasMPI, uniform_force_and_virial)
{
    add_core();
    lammps_command(handle, "fix early all addforce 0.25 0.5 -0.125");
    auto *bias = add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    const auto result = forces();
    const std::array<double, 3> combined{0.375, 0.25, 0.25};
    for (int atom = 0; atom < 2; ++atom)
        for (int d = 0; d < 3; ++d)
            EXPECT_NEAR(result[3 * atom + d], rank / 2 == 0 ? std::sqrt(2.0) * combined[d] : 0.0,
                        1.0e-12);
    EXPECT_EQ(bias->completed, 1);
    std::array<double, 6> actual_virial{};
    MPI_Allreduce(bias->virial, actual_virial.data(), 6, MPI_DOUBLE, MPI_SUM,
                  static_cast<LAMMPS *>(handle)->world);
    const double sumx = rank / 2 == 0 ? 20.0 : 11.0;
    const double sumy = rank / 2 == 0 ? 20.0 : 24.0;
    const std::array<double, 6> expected{sumx * 0.125, sumy * -0.25, 4.0 * 0.375,
                                         sumy * 0.125, 4.0 * 0.125,  4.0 * -0.25};
    for (int d = 0; d < 6; ++d)
        EXPECT_NEAR(actual_virial[d], expected[d], 1.0e-12);
    EXPECT_NEAR(bias->physical_energy, -(sumx * 0.125 + sumy * -0.25 + 4.0 * 0.375) / 2.0, 1.0e-12);
}

TEST_F(PIMDBiasMPI, repeated_same_step_generation_and_energy)
{
    add_core();
    auto *bias = add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    const auto first_token = bias->registration;
    const double before    = bias->physical_energy;
    lammps_command(handle, "displace_atoms all move 0.5 0.0 0.0 units box");
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    ASSERT_EQ(bias->generations.size(), 2);
    EXPECT_NE(bias->generations[0], bias->generations[1]);
    EXPECT_EQ(bias->timesteps[0], bias->timesteps[1]);
    EXPECT_EQ(first_token, bias->registration);
    EXPECT_EQ(bias->completed, 2);
    EXPECT_NEAR(bias->physical_energy - before, -0.0625, 1.0e-12);
}

TEST_F(PIMDBiasMPI, duplicate_submission)
{
    add_core();
    auto *bias     = add_bias();
    bias->behavior = MockBehavior::DUPLICATE;
    expect_command_error("run 0 post no", "Duplicate external PIMD bias force submission");
    EXPECT_EQ(bias->completed, 1);
    const auto result = forces();
    for (int atom = 0; atom < 2; ++atom)
        for (int d = 0; d < 3; ++d)
            EXPECT_NEAR(result[3 * atom + d],
                        rank / 2 == 0 ? std::sqrt(2.0) * bias->increment[d] : 0.0, 1.0e-12);
}

TEST_F(PIMDBiasMPI, stale_evaluation)
{
    add_core();
    auto *bias = add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    bias->behavior = MockBehavior::STALE;
    expect_command_error("run 0 post no", "Stale external PIMD bias force evaluation");
    EXPECT_EQ(bias->completed, 1);
}

TEST_F(PIMDBiasMPI, missing_submission_run_zero)
{
    add_core();
    auto *bias     = add_bias();
    bias->behavior = MockBehavior::OMIT;
    expect_command_error("run 0 post no", "Missing external PIMD bias force submission");
    EXPECT_EQ(bias->completed, 0);
}

TEST_F(PIMDBiasMPI, wrong_atom_count)
{
    add_core();
    auto *bias     = add_bias();
    bias->behavior = MockBehavior::WRONG_COUNT;
    expect_command_error("run 0 post no", "Invalid external PIMD bias submission");
    EXPECT_EQ(bias->completed, 0);
}

TEST_F(PIMDBiasMPI, second_provider_rejected)
{
    add_core();
    auto *first = add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    auto *second = add_bias("second_bias");
    auto *api    = interface();
    EXPECT_THROW(api->register_bias(second, PIMDBiasMode::COORDINATE_MEAN, false), LAMMPSException);
    EXPECT_EQ(first->registration->owner, api);
    lammps_command(handle, "unfix second_bias");
    lammps_command(handle, "run 0 post no");
    EXPECT_EQ(lammps_has_error(handle), 0);
    EXPECT_EQ(first->completed, 2);
}

TEST_F(PIMDBiasMPI, post_force_after_provider_rejected)
{
    add_core();
    add_bias();
    lammps_command(handle, "fix late all addforce 0.25 0.0 0.0");
    expect_command_error(
        "run 0 post no",
        "External PIMD bias adapter must be defined after all other post-force fixes");
}

TEST_F(PIMDBiasMPI, core_after_provider_rejected)
{
    add_bias();
    add_core();
    expect_command_error("run 0 post no",
                         "External PIMD bias adapter must be defined after its integrator");
    // Failed registration must not leave a dangling provider in the core.
    lammps_command(handle, "unfix bias");
    add_bias();
    lammps_command(handle, "run 0 post no");
    EXPECT_EQ(lammps_has_error(handle), 0);
}

TEST_F(PIMDBiasMPI, unregister_and_recreate)
{
    add_core();
    auto *bias = add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    const auto old_registration = bias->registration;
    lammps_command(handle, "unfix bias");
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    for (double value : forces())
        EXPECT_NEAR(value, 0.0, 1.0e-12);
    bias = add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    EXPECT_NE(old_registration, bias->registration);
    EXPECT_EQ(bias->completed, 1);
    auto *api = interface();
    EXPECT_THROW(api->bias_coordinates(*old_registration), LAMMPSException);
}

TEST_F(PIMDBiasMPI, core_destroyed_before_adapter)
{
    add_core();
    auto *bias = add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    const auto registration = bias->registration;
    lammps_command(handle, "unfix fpimd");
    ASSERT_EQ(lammps_has_error(handle), 0);
    EXPECT_EQ(registration->owner, nullptr);
    lammps_command(handle, "unfix bias");
    EXPECT_EQ(lammps_has_error(handle), 0);
}

TEST_F(PIMDBiasMPI, stale_generation_on_empty_rank)
{
    add_core();
    auto *bias = add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    if (rank == 3) EXPECT_EQ(static_cast<LAMMPS *>(handle)->atom->nlocal, 0);
    bias->behavior = MockBehavior::STALE_ONE_RANK;
    expect_command_error("run 0 post no", "Stale external PIMD bias force evaluation");
    EXPECT_EQ(bias->completed, 1);
}

TEST_F(PIMDBiasMPI, direct_spring_once)
{
    add_core("pimd");
    auto *bias                                         = add_bias();
    bias->increment                                    = {0.0, 0.0, 0.0};
    const std::array<std::array<double, 6>, 2> initial = {
        {{5.0, 10.0, 2.0, 15.0, 10.0, 2.0}, {5.0, 10.0, 2.0, 6.0, 14.0, 2.0}}};
    const double omega_p   = 4.0 * std::acos(-1.0);
    const double stiffness = 2.0 * omega_p * omega_p;
    constexpr double dt    = 0.0001;
    for (int setup = 0; setup < 2; ++setup) {
        lammps_command(handle, "run 0 post no");
        ASSERT_EQ(lammps_has_error(handle), 0);
        const auto actual = state();
        for (int bead = 0; bead < 2; ++bead)
            for (int d = 0; d < 6; ++d)
                EXPECT_NEAR(actual[18 * bead + 12 + d],
                            stiffness * (initial[1 - bead][d] - initial[bead][d]), 1.0e-10);
    }
    lammps_command(handle, "run 1 pre no post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    const auto actual = state();
    for (int bead = 0; bead < 2; ++bead) {
        for (int d = 0; d < 6; ++d) {
            const double f0     = stiffness * (initial[1 - bead][d] - initial[bead][d]);
            const double x1     = initial[bead][d] + 0.5 * dt * dt * f0;
            const double other1 = initial[1 - bead][d] - 0.5 * dt * dt * f0;
            const double f1     = stiffness * (other1 - x1);
            EXPECT_NEAR(actual[18 * bead + d], x1, 1.0e-12);
            EXPECT_NEAR(actual[18 * bead + 6 + d], 0.5 * dt * (f0 + f1), 1.0e-12);
            EXPECT_NEAR(actual[18 * bead + 12 + d], f1, 1.0e-10);
        }
    }
    EXPECT_EQ(bias->completed, 3);
}

TEST_F(PIMDBiasMPI, hook_after_transform_before_final_kick)
{
    add_core("nmpimd", "PIMD/TEST/CORE");
    auto *bias = add_bias();
    auto *core = static_cast<FixTestPIMDCore *>(
        static_cast<LAMMPS *>(handle)->modify->get_fix_by_id("fpimd"));
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    EXPECT_EQ(core->hook_calls, 1);
    lammps_command(handle, "run 1 pre no post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    EXPECT_EQ(core->hook_calls, 2);
    EXPECT_EQ(core->hooks_before_final_kick, 2);
    EXPECT_EQ(core->final_calls, 1);
    EXPECT_EQ(bias->completed, 2);
    for (const auto &force : core->hook_force)
        for (int d = 0; d < 3; ++d)
            EXPECT_NEAR(force[d], rank / 2 == 0 ? std::sqrt(2.0) * bias->increment[d] : 0.0,
                        1.0e-12);
}

TEST_F(PIMDBiasMPI, split_and_restart_continuation)
{
    add_core();
    add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    write_checkpoint("test_pimd_mock_initial");
    lammps_command(handle, "run 3 pre no post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    const auto continuous = state();

    read_checkpoint("test_pimd_mock_initial");
    lammps_command(handle, "run 1 post no");
    lammps_command(handle, "run 2 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    const auto split = state();

    read_checkpoint("test_pimd_mock_initial");
    lammps_command(handle, "run 1 post no");
    write_checkpoint("test_pimd_mock_split");
    read_checkpoint("test_pimd_mock_split");
    lammps_command(handle, "run 2 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    const auto restarted = state();
    for (std::size_t i = 0; i < continuous.size(); ++i) {
        EXPECT_NEAR(split[i], continuous[i], 1.0e-12) << i;
        EXPECT_NEAR(restarted[i], continuous[i], 1.0e-12) << i;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (const char *prefix : {"test_pimd_mock_initial", "test_pimd_mock_split"})
            for (int bead = 0; bead < 2; ++bead)
                std::remove((std::string(prefix) + "." + std::to_string(bead)).c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(PIMDBiasMPI, capacity_growth_during_migration)
{
    // create_box starts with one slot. Reserve a small real AtomVec capacity
    // before creating atoms, then let the normal MPI exchange grow it in-run.
    lammps_command(handle, "clear");
    for (const char *command :
         {"units lj", "atom_style atomic", "atom_modify map array sort 0 0.0", "processors 2 1 1",
          "region box block 0 20 0 20 0 20", "create_box 1 box"})
        lammps_command(handle, command);
    auto *lmp = static_cast<LAMMPS *>(handle);
    lmp->atom->avec->grow(8);
    constexpr int natoms      = 12;
    constexpr double dt       = 0.0001;
    constexpr double strength = 0.125;
    std::array<std::array<double, 3>, natoms> initial{};
    for (int i = 0; i < natoms; ++i) {
        initial[i] = {i < 6 ? 6.0 + 0.125 * i : 11.0 + 0.125 * (i - 6), 4.0 + 0.25 * (i % 3),
                      2.0 + 0.125 * (i % 2)};
        const std::string command = "create_atoms 1 single " + std::to_string(initial[i][0]) + " " +
                                    std::to_string(initial[i][1]) + " " +
                                    std::to_string(initial[i][2]);
        lammps_command(handle, command.c_str());
    }
    for (const char *command :
         {"mass 1 1.0", "pair_style zero 0.1", "pair_coeff * *", "neighbor 0.01 nsq",
          "neigh_modify every 1 delay 0 check no", "group left id 1:6", "group right id 7:12",
          "variable left_v world 0.0 45000.0", "variable right_v world -20000.0 0.0",
          "velocity left set ${left_v} 0.0 0.0 units box",
          "velocity right set ${right_v} 0.0 0.0 units box", "timestep 0.0001",
          "thermo_style custom step", "thermo 0"})
        lammps_command(handle, command);
    lammps_set_show_error(handle, 0);
    const auto all_true = [](bool condition) {
        int result = condition ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return result != 0;
    };
    ASSERT_TRUE(all_true(lammps_has_error(handle) == 0));
    ASSERT_TRUE(all_true((lmp->atom->nlocal == 6) && (lmp->atom->nmax == 8)));
    add_core("pimd");
    auto *bias = add_bias();
    ASSERT_TRUE(all_true(bias != nullptr));
    bias->harmonic_bias = bias->capture_coordinates = true;
    bias->harmonic_strength                         = strength;
    lammps_command(handle, "run 0 post no");
    ASSERT_TRUE(all_true(lammps_has_error(handle) == 0));
    ASSERT_TRUE(all_true((lmp->atom->nlocal == 6) && (lmp->atom->nmax == 8)));
    const int old_nmax      = lmp->atom->nmax;
    const auto registration = bias->registration;
    const double stiffness  = 2.0 * std::pow(4.0 * std::acos(-1.0), 2);
    const auto position     = [&](int atom, int bead, int step) {
        auto result = initial[atom];
        if (step) {
            const double vx = atom < 6 ? (bead == 1 ? 45000.0 : 0.0) : (bead == 0 ? -20000.0 : 0.0);
            for (int d = 0; d < 3; ++d)
                result[d] += (d == 0 ? dt * vx : 0.0) - 0.5 * dt * dt * strength * initial[atom][d];
        }
        return result;
    };
    const auto verify = [&](int step) {
        const int bead = rank / 2;
        bool valid     = bias->observations.size() == static_cast<std::size_t>(lmp->atom->nlocal);
        for (const auto &observed : bias->observations)
            valid = valid && (observed.tag >= 1) && (observed.tag <= natoms);
        ASSERT_TRUE(all_true(valid));
        std::array<double, 6 * natoms> actual_force{};
        std::array<int, 2 * natoms> owners{};
        for (int i = 0; i < lmp->atom->nlocal; ++i) {
            const auto &observed = bias->observations[i];
            const int atom       = static_cast<int>(observed.tag) - 1;
            const auto own       = position(atom, bead, step);
            const auto other     = position(atom, 1 - bead, step);
            EXPECT_EQ(observed.tag, lmp->atom->tag[i]);
            ++owners[bead * natoms + atom];
            for (int d = 0; d < 3; ++d) {
                EXPECT_NEAR(observed.bead[d], own[d], 1.0e-12);
                EXPECT_NEAR(observed.unwrapped[d], own[d], 1.0e-12);
                EXPECT_NEAR(observed.centroid[d], 0.5 * (own[d] + other[d]), 1.0e-12);
                EXPECT_NEAR(observed.force[d], -0.5 * strength * own[d], 1.0e-12);
                EXPECT_NEAR(lmp->atom->x[i][d], own[d], 1.0e-12);
                EXPECT_NEAR(lmp->atom->f[i][d],
                            -strength * own[d] + stiffness * (other[d] - own[d]), 1.0e-10);
                actual_force[3 * (bead * natoms + atom) + d] = lmp->atom->f[i][d];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, owners.data(), static_cast<int>(owners.size()), MPI_INT,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, actual_force.data(), static_cast<int>(actual_force.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int owner : owners)
            EXPECT_EQ(owner, 1);
        for (int atom = 0; atom < natoms; ++atom) {
            const auto first  = position(atom, 0, step);
            const auto second = position(atom, 1, step);
            for (int d = 0; d < 3; ++d)
                EXPECT_NEAR(actual_force[3 * atom + d] + actual_force[3 * (natoms + atom) + d],
                            -strength * (first[d] + second[d]), 1.0e-10);
        }
        double energy = 0.0;
        std::array<double, 6> expected_virial{}, actual_virial{};
        constexpr std::array<int, 6> rows{0, 1, 2, 0, 0, 1};
        constexpr std::array<int, 6> columns{0, 1, 2, 1, 2, 2};
        for (int atom = 0; atom < natoms; ++atom) {
            const auto x = position(atom, bead, step);
            for (int d = 0; d < 3; ++d)
                energy += 0.25 * strength * x[d] * x[d];
            for (int d = 0; d < 6; ++d)
                expected_virial[d] -= strength * x[rows[d]] * x[columns[d]];
        }
        MPI_Allreduce(bias->virial, actual_virial.data(), 6, MPI_DOUBLE, MPI_SUM, lmp->world);
        EXPECT_NEAR(bias->physical_energy, energy, 1.0e-10);
        for (int d = 0; d < 6; ++d)
            EXPECT_NEAR(actual_virial[d], expected_virial[d], 1.0e-10);
    };
    verify(0);

    // Opposite bead ownership exercises both local storage growth and requests
    // larger than an empty partner's original inter-replica send capacity.
    lammps_command(handle, "run 1 pre no post no");
    ASSERT_TRUE(all_true(lammps_has_error(handle) == 0));
    const bool receiver = (rank == 0) || (rank == 3);
    ASSERT_TRUE(all_true(lmp->atom->nlocal == (receiver ? natoms : 0)));
    ASSERT_TRUE(
        all_true(!receiver || ((lmp->atom->nlocal > old_nmax) && (lmp->atom->nmax > old_nmax))));
    verify(1);
    EXPECT_EQ(bias->completed, 2);
    EXPECT_EQ(bias->registration, registration);

    lammps_command(handle, "run 0 post no");
    ASSERT_TRUE(all_true(lammps_has_error(handle) == 0));
    verify(1);
    ASSERT_TRUE(all_true(bias->generations.size() == 3));
    EXPECT_LT(bias->generations[0], bias->generations[1]);
    EXPECT_LT(bias->generations[1], bias->generations[2]);
    EXPECT_EQ(bias->timesteps[1], bias->timesteps[2]);
    EXPECT_EQ(bias->completed, 3);
    EXPECT_EQ(bias->registration, registration);
}

TEST_F(PIMDBiasMPI, physical_bias_finite_strain_virial)
{
    add_core();
    auto *bias = add_bias();
    lammps_command(handle, "run 0 post no");
    ASSERT_EQ(lammps_has_error(handle), 0);
    auto *lmp = static_cast<LAMMPS *>(handle);
    std::vector<std::array<double, 3>> reference(lmp->atom->nlocal);
    for (int i = 0; i < lmp->atom->nlocal; ++i)
        for (int d = 0; d < 3; ++d)
            reference[i][d] = lmp->atom->x[i][d];
    std::array<double, 6> dynamical_virial{};
    MPI_Allreduce(bias->virial, dynamical_virial.data(), 6, MPI_DOUBLE, MPI_SUM, lmp->world);
    // Apply x_row -> x_row + epsilon*x_column.  This checks the native
    // xx, yy, zz, xy, xz, yz strain convention using actual bias energies.
    constexpr std::array<int, 6> rows{0, 1, 2, 0, 0, 1};
    constexpr std::array<int, 6> columns{0, 1, 2, 1, 2, 2};
    constexpr double epsilon = 1.0e-5;
    for (int component = 0; component < 6; ++component) {
        SCOPED_TRACE(component);
        std::array<double, 2> energy{};
        for (int sign = 0; sign < 2; ++sign) {
            for (int i = 0; i < lmp->atom->nlocal; ++i) {
                for (int d = 0; d < 3; ++d)
                    lmp->atom->x[i][d] = reference[i][d];
                lmp->atom->x[i][rows[component]] +=
                    (sign == 0 ? epsilon : -epsilon) * reference[i][columns[component]];
            }
            lammps_command(handle, "run 0 post no");
            ASSERT_EQ(lammps_has_error(handle), 0);
            energy[sign] = bias->physical_energy;
        }
        const double physical_derivative = (energy[0] - energy[1]) / (2.0 * epsilon);
        EXPECT_NEAR(dynamical_virial[component], -2.0 * physical_derivative, 1.0e-9);
    }
}

TEST_F(PIMDBiasMPI, harmonic_bias_preserves_physical_estimators)
{
    for (const char *method : {"pimd", "nmpimd"}) {
        SCOPED_TRACE(method);
        add_core(method);
        auto *bias              = add_bias();
        bias->harmonic_bias     = true;
        bias->harmonic_strength = 0.0;
        lammps_command(handle, "run 0 post no");
        ASSERT_EQ(lammps_has_error(handle), 0);
        auto *core                = static_cast<LAMMPS *>(handle)->modify->get_fix_by_id("fpimd");
        const double initial_tvir = core->compute_vector(5);
        const double initial_tcv  = core->compute_vector(6);
        EXPECT_DOUBLE_EQ(initial_tvir, 0.0);
        EXPECT_DOUBLE_EQ(initial_tcv, 3.0);

        bias->harmonic_strength = 0.125;
        lammps_command(handle, "run 0 post no");
        ASSERT_EQ(lammps_has_error(handle), 0);
        EXPECT_DOUBLE_EQ(core->compute_vector(5), initial_tvir);
        EXPECT_DOUBLE_EQ(core->compute_vector(6), initial_tcv);
        // The physical pair force is zero; biased cvir is nonzero because
        // the second atom has different bead coordinates.  With zero KE,
        // p_cv = -k sum_i |R_i|^2/(3V), independent of the bead spread.
        constexpr double centroid_square =
            5.0 * 5.0 + 10.0 * 10.0 + 2.0 * 2.0 + 10.5 * 10.5 + 12.0 * 12.0 + 2.0 * 2.0;
        EXPECT_NEAR(core->compute_vector(9),
                    -bias->harmonic_strength * centroid_square / (3.0 * 8000.0), 1.0e-14);
        lammps_command(handle, "unfix bias");
        lammps_command(handle, "unfix fpimd");
    }
}

TEST_F(PIMDBiasMPI, harmonic_bias_refreshes_anisotropic_stress)
{
    constexpr std::array<int, 6> rows{0, 1, 2, 0, 0, 1};
    constexpr std::array<int, 6> columns{0, 1, 2, 1, 2, 2};
    constexpr std::array<std::array<double, 3>, 2> centroids{{{5.0, 10.0, 2.0}, {10.5, 12.0, 2.0}}};
    for (const char *ensemble : {"nph", "npt"}) {
        SCOPED_TRACE(ensemble);
        const std::string command =
            "fix fpimd all PIMD/TEST/CORE method nmpimd ensemble " + std::string(ensemble) +
            " integrator obabo thermostat PILE_L 1234 tau 1.0e100 "
            "temp 1.0 sp 1.0 fmass 1.0 x 0.0 y 0.0 z 0.0 barostat BZP taup 1.0 fixcom no";
        lammps_command(handle, command.c_str());
        auto *bias              = add_bias();
        bias->harmonic_bias     = true;
        bias->harmonic_strength = 0.0;
        lammps_command(handle, "run 0 post no");
        ASSERT_EQ(lammps_has_error(handle), 0);
        auto *core = static_cast<FixTestPIMDCore *>(
            static_cast<LAMMPS *>(handle)->modify->get_fix_by_id("fpimd"));
        for (double value : core->current_stress())
            EXPECT_DOUBLE_EQ(value, 0.0);

        bias->harmonic_strength = 0.125;
        lammps_command(handle, "run 0 post no");
        ASSERT_EQ(lammps_has_error(handle), 0);
        const auto stress = core->current_stress();
        double trace      = 0.0;
        for (int component = 0; component < 6; ++component) {
            double expected = 0.0;
            for (const auto &centroid : centroids)
                expected -= bias->harmonic_strength * centroid[rows[component]] *
                            centroid[columns[component]] / 8000.0;
            EXPECT_NEAR(stress[component], expected, 1.0e-14) << component;
            if (component < 3) trace += expected;
        }
        EXPECT_NEAR(core->compute_vector(9), trace / 3.0, 1.0e-14);
        EXPECT_DOUBLE_EQ(core->compute_vector(5), 0.0);
        EXPECT_DOUBLE_EQ(core->compute_vector(6), 3.0);
        lammps_command(handle, "unfix bias");
        lammps_command(handle, "unfix fpimd");
    }
}

TEST_F(PIMDBiasMPI, physical_energy_without_global_scalar_tally)
{
    add_core();
    auto *bias          = add_bias();
    bias->harmonic_bias = true;
    auto *core          = static_cast<LAMMPS *>(handle)->modify->get_fix_by_id("fpimd");
    // Submission is the source of physical B even when the adapter does not
    // contribute a global scalar to compute pe.  The dynamical energy is P*B
    // for every native global-energy/thermo flag combination.
    for (int global_energy : {0, 1}) {
        for (int thermo_energy : {0, 1}) {
            SCOPED_TRACE(global_energy);
            SCOPED_TRACE(thermo_energy);
            bias->energy_global_flag = global_energy;
            bias->thermo_energy      = thermo_energy;
            lammps_command(handle, "run 0 post no");
            ASSERT_EQ(lammps_has_error(handle), 0);
            EXPECT_NEAR(core->compute_vector(2), 2.0 * bias->physical_energy, 1.0e-12);
            EXPECT_GT(bias->physical_energy, 0.0);
        }
    }
}

} // namespace
