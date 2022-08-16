#include <iostream>
#include <iomanip>
#include <benchmark/benchmark.h>
#include "../include/vec3i128.h"

using namespace std;

#define pi 3.141592653589793
#define solar_mass (4 * pi * pi)
#define days_per_year 365.24

#define NBODIES 5

const int fixed128_fraction_bits = 40;
const double fixed_multiplier = std::pow(2.0, fixed128_fraction_bits);
const double fixed_multiplier_inv = 1 / fixed_multiplier;

struct planet {
    int128 x, y, z;
    double vx, vy, vz;
    double mass;
};

void advance(int nbodies, struct planet* bodies, double dt)
{
    int i, j;

    for (i = 0; i < nbodies; i++) {
        struct planet* b = &(bodies[i]);
        for (j = i + 1; j < nbodies; j++) {
            struct planet* b2 = &(bodies[j]);
            double dx = sub_to_double(b->x, b2->x) * fixed_multiplier_inv;
            double dy = sub_to_double(b->y, b2->y) * fixed_multiplier_inv;
            double dz = sub_to_double(b->z, b2->z) * fixed_multiplier_inv;
            double distance = sqrt(dx * dx + dy * dy + dz * dz);
            double mag = dt / (distance * distance * distance);
            b->vx -= dx * b2->mass * mag;
            b->vy -= dy * b2->mass * mag;
            b->vz -= dz * b2->mass * mag;
            b2->vx += dx * b->mass * mag;
            b2->vy += dy * b->mass * mag;
            b2->vz += dz * b->mass * mag;
        }
    }
    for (i = 0; i < nbodies; i++) {
        struct planet* b = &(bodies[i]);
        b->x += (dt * b->vx) * fixed_multiplier;
        b->y += (dt * b->vy) * fixed_multiplier;
        b->z += (dt * b->vz) * fixed_multiplier;
    }
}

double energy(int nbodies, struct planet* bodies)
{
    double e;
    int i, j;

    e = 0.0;
    for (i = 0; i < nbodies; i++) {
        struct planet* b = &(bodies[i]);
        e += 0.5 * b->mass * (b->vx * b->vx + b->vy * b->vy + b->vz * b->vz);
        for (j = i + 1; j < nbodies; j++) {
            struct planet* b2 = &(bodies[j]);
            double dx = sub_to_double(b->x, b2->x) * fixed_multiplier_inv;
            double dy = sub_to_double(b->y, b2->y) * fixed_multiplier_inv;
            double dz = sub_to_double(b->z, b2->z) * fixed_multiplier_inv;
            double distance = sqrt(dx * dx + dy * dy + dz * dz);
            e -= (b->mass * b2->mass) / distance;
        }
    }
    return e;
}

void offset_momentum(int nbodies, struct planet* bodies)
{
    double px = 0.0, py = 0.0, pz = 0.0;
    int i;
    for (i = 0; i < nbodies; i++) {
        px += bodies[i].vx * bodies[i].mass;
        py += bodies[i].vy * bodies[i].mass;
        pz += bodies[i].vz * bodies[i].mass;
    }
    bodies[0].vx = -px / solar_mass;
    bodies[0].vy = -py / solar_mass;
    bodies[0].vz = -pz / solar_mass;
}

double energyStart;
double energyEnd;

static void NBody_I128(benchmark::State& state)
{
    int n = 50000000;
    planet bodies[NBODIES]{
    {                               /* sun */
        0.0, 0.0, 0.0, 0, 0, 0, solar_mass
    },
    {                               /* jupiter */
        (int128)( 4.84143144246472090e+00 * fixed_multiplier),
        (int128)(-1.16032004402742839e+00 * fixed_multiplier),
        (int128)(-1.03622044471123109e-01 * fixed_multiplier),
        1.66007664274403694e-03 * days_per_year,
        7.69901118419740425e-03 * days_per_year,
        -6.90460016972063023e-05 * days_per_year,
        9.54791938424326609e-04 * solar_mass
    },
  {                               /* saturn */
    (int128)( 8.34336671824457987e+00 * fixed_multiplier),
    (int128)( 4.12479856412430479e+00 * fixed_multiplier),
    (int128)(-4.03523417114321381e-01 * fixed_multiplier),
    -2.76742510726862411e-03 * days_per_year,
    4.99852801234917238e-03 * days_per_year,
    2.30417297573763929e-05 * days_per_year,
    2.85885980666130812e-04 * solar_mass
  },
  {                               /* uranus */
    (int128)( 1.28943695621391310e+01 * fixed_multiplier),
    (int128)(-1.51111514016986312e+01 * fixed_multiplier),
    (int128)(-2.23307578892655734e-01 * fixed_multiplier),
    2.96460137564761618e-03 * days_per_year,
    2.37847173959480950e-03 * days_per_year,
    -2.96589568540237556e-05 * days_per_year,
    4.36624404335156298e-05 * solar_mass
  },
  {                               /* neptune */
    (int128)( 1.53796971148509165e+01 * fixed_multiplier),
    (int128)(-2.59193146099879641e+01 * fixed_multiplier),
    (int128)( 1.79258772950371181e-01 * fixed_multiplier),
    2.68067772490389322e-03 * days_per_year,
    1.62824170038242295e-03 * days_per_year,
    -9.51592254519715870e-05 * days_per_year,
    5.15138902046611451e-05 * solar_mass
  }
    };
    for (auto _ : state)
    {
        offset_momentum(NBODIES, bodies);
        energyStart = energy(NBODIES, bodies);
        for (int i = 0; i < n; i++)
            advance(NBODIES, bodies, 0.01);
        energyEnd = energy(NBODIES, bodies);
    }
}

BENCHMARK(NBody_I128);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    std::cout << std::setprecision(10) << energyStart << std::endl;
    std::cout << std::setprecision(10) << energyEnd << std::endl;
    std::cout << std::setprecision(10) << "Energy error:" << energyEnd - energyStart << std::endl;
    return 0;
}