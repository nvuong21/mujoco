// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_SRC_ENGINE_ENGINE_COLLISION_DRIVER_H_
#define MUJOCO_SRC_ENGINE_ENGINE_COLLISION_DRIVER_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>

#ifdef __cplusplus
extern "C" {
#endif

// collision function pointers and max contact pairs
MJAPI extern mjfCollision mjCOLLISIONFUNC[mjNGEOMTYPES][mjNGEOMTYPES];

// collision detection entry point
MJAPI void mj_collision(const mjModel* m, mjData* d);

// broad phase collistion detection; return list of body pairs for narrow phase
int mj_broadphase(const mjModel* m, mjData* d, int* bodypair, int maxpair);

// test two geoms for collision, apply filters, add to contact list
//  flg_user disables filters and uses usermargin
// this function calls mj_collideGeomsSkip
void mj_collideGeoms(const mjModel* m, mjData* d,
                     int g1, int g2, int flg_user, mjtNum usermargin);

// NOTE remove these functions, process directly data.contact
// // test two geoms for collision, apply filters, but not add to contact list yet
// int mj_collideGeomsSkip(const mjModel* m, mjData* d, mjContact* con,
//                      int g1, int g2, int flg_user, mjtNum usermargin);

// // add a list of contacts to mjData
// void mj_addContacts(const mjModel* m, mjData* d, const mjContact* con, int ncon);

// perform contact clustering on a list of contacts
// cluster a set of contacts in d->contact from start_idx to end_idx.
// If end_idx=-1 then cluster from start to data->ncon
void mj_clusterContacts(const mjModel* m, mjData* d, int start_idx, int end_idx);
// int mj_clusterContacts(mjData* d, mjContact* res, const mjContact* con, int ncon);

// number of possible collisions based on fitlers and geom types
int mj_contactFilter(int type1, int contype1, int conaffinity1, int weldbody1, int weldparent1,
                     int type2, int contype2, int conaffinity2, int weldbody2, int weldparent2,
                     int filterparent);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_DRIVER_H_
