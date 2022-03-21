add_library(models
    double_integrator.h
    double_integrator.c
)

target_link_libraries(models
  PUBLIC
  riccati
  slap
)

add_target_to_install(models)

