// -*- C++ -*-

#include "random.hpp"

#include "catch.hpp"

using namespace nix;

inline void compare_with_reference(float64 temperature, float64 drift, int size,
                                   double reference[][3])
{
  MaxwellJuttner  mj(temperature, drift);
  std::mt19937_64 random(0);

  for (int i = 0; i < size; ++i) {
    auto [ux, uy, uz] = mj(random);

    REQUIRE(std::abs(ux - reference[i][0]) < 1e-14);
    REQUIRE(std::abs(uy - reference[i][1]) < 1e-14);
    REQUIRE(std::abs(uz - reference[i][2]) < 1e-14);
  }
}

inline void generate_and_save(float64 temperature, float64 drift, int size, std::string filename)
{
  MaxwellJuttner  mj(temperature, drift);
  std::mt19937_64 random(0);

  std::ofstream ofs(filename, std::ios::binary);

  for (int i = 0; i < size; i++) {
    auto [ux, uy, uz] = mj(random);

    ofs.write(reinterpret_cast<const char*>(&ux), sizeof(ux));
    ofs.write(reinterpret_cast<const char*>(&uy), sizeof(uy));
    ofs.write(reinterpret_cast<const char*>(&uz), sizeof(uz));
  }

  ofs.close();
}

TEST_CASE("MaxwellJuttner")
{
  constexpr int size = 10;

  SECTION("(1) temperature = 0.5, drift = 1.0")
  {
    float64 temperature = 0.5;
    float64 drift       = 1.0;

    double reference[size][3] = {
        {+2.252512876784235e+00, +5.588982554705004e-01, -3.731828861547884e-01},
        {+2.207124613005604e+00, -2.688199810217650e-01, -7.529104474631365e-01},
        {+2.047005711525658e-01, +1.955107685914205e+00, +2.461629446415522e-01},
        {+4.687805644230362e-01, +3.537058723228913e-01, +8.918507600549205e-01},
        {+3.389096415691068e+00, +5.457064020495481e-01, +1.031561824316754e+00},
        {+1.051836102659536e+00, -6.688670612418387e-01, -6.721419110951746e-02},
        {+1.669334971110022e+00, +5.322478689522300e-01, -8.051292168647054e-01},
        {+2.077729736429222e+00, -1.257660183861676e-01, +6.986721954693282e-01},
        {+5.719549199804868e-01, +7.080367241181165e-01, +2.168111958964141e-01},
        {-9.670836244444861e-01, -6.571604688736042e-02, +4.225611191910858e-01},
    };

    compare_with_reference(temperature, drift, size, reference);
  }

  SECTION("(2) temperature = 2.0, drift = 0.2")
  {
    float64 temperature = 2.0;
    float64 drift       = 0.2;

    double reference[size][3] = {
        {+3.079204458762005e-01, -3.747302344580209e+00, -5.553368051876226e+00},
        {+3.497825730362293e+00, -2.542203403373179e+00, +1.060946378474007e+01},
        {+1.859102875867483e+00, -2.259480617321652e+00, -5.438641225674193e+00},
        {+9.131756760814060e-01, -2.633649513270879e-01, +5.908445667568254e+00},
        {-5.108663717340580e+00, +5.657875996298354e+00, +6.027308583362071e+00},
        {+2.105115359836829e+00, -8.997276036956675e+00, +2.802328771361430e-01},
        {+4.023814152517910e+00, +2.123837012348138e+00, -2.443006343328988e+00},
        {+5.107237042414590e+00, +5.819542122881540e-01, -1.889300823854473e+00},
        {+7.470885615700431e+00, -2.886429980646654e+00, -7.737142140541478e+00},
        {+2.729425245528705e+00, -7.529228263765149e-01, +1.361115401425296e+00},
    };

    compare_with_reference(temperature, drift, size, reference);
  }
}

//
// This test is prepared for saving generated random numbers to files.
// The saved data can be used for plotting the distribution to compare with
// the analytic distribution.
//
// By default, this test is disabled.
// To execute this test, run the test with the following command:
//
//   $ ./test_random "[.SaveData]"
//
TEST_CASE("GenerateMaxwellJuttner", "[.SaveData]")
{
  constexpr int size = 100000;

  SECTION("(1) temperature = 0.5, drift = 1.0")
  {
    float64 temperature = 0.5;
    float64 drift       = 1.0;

    generate_and_save(temperature, drift, size, "test_random1.dat");
  }

  SECTION("(2) temperature = 2.0, drift = 0.2")
  {
    float64 temperature = 2.0;
    float64 drift       = 0.2;

    generate_and_save(temperature, drift, size, "test_random2.dat");
  }
}
