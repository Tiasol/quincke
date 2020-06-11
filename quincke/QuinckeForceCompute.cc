// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "QuinckeForceCompute.h"
#ifdef ENABLE_CUDA
#include "QuinckeForceCompute.cuh"
#endif

/*! \file ExampleUpdater.cc
    \brief Definition of ExampleUpdater
*/

// ********************************
// here follows the code for ExampleUpdater on the CPU

/*! \param sysdef System to zero the velocities of
*/
QuinckeForceCompute::QuinckeForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<NeighborList> nlist,
                                        std::shared_ptr<ParticleGroup> group,
                                        Scalar Dpassive,
                                        Scalar Dactive,
                                        Scalar Ee,
                                        Scalar Ecut,
                                        Scalar rcut,
                                        Scalar sigma21,
                                        Scalar H,
                                        Scalar epsilon,
                                        int seed)
        : ForceCompute(sysdef), m_nlist(nlist), m_group(group), m_Dpassive(Dpassive), 
        m_Dactive(Dactive), m_Ee(Ee), m_Ecut(Ecut), m_rcut(rcut), m_sigma21(sigma21), 
        m_H(H), m_epsilon(epsilon)
    {

    assert(m_nlist);
    assert(m_pdata);

    unsigned int group_size = m_group->getNumMembersGlobal();
    if (group_size == 0)
        {
        m_exec_conf->msg->error() << "Creating a QuinckeForceCompute with an empty group" << std::endl;
        throw std::runtime_error("Error initializing QuinckeForceCompute");
        }

    last_computed = 10;

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = seed*0x12345677 + 0x12345; seed^=(seed>>16); seed*= 0x45679;

    // broadcast the seed from rank 0 to all other ranks.
    #ifdef ENABLE_MPI
        if(this->m_pdata->getDomainDecomposition())
            bcast(m_seed, 0, this->m_exec_conf->getMPICommunicator());
    #endif
    
    }


QuinckeForceCompute::~QuinckeForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ActiveForceCompute" << std::endl;
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void QuinckeForceCompute::setForces(unsigned int timestep)
    {

    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push("QuinckeForceCompute");

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    // bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // // array handles
    // access neighbor list
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);
    // access particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);


    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::overwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();


    // sanity check
    assert(h_orientation.data != NULL);

    // need to start from a zero forces
    memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset(h_torque.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    
    
    // for each particle
    for (int i = 0; i < m_group->getNumMembers(); i++)
        {
        

        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];

        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        // unsigned int typei = __scalar_as_int(h_pos.data[idx].w);
        // quat<Scalar> quati(h_orientation.data[idx]);
        
        
        // cal radius
        Scalar ai = h_diameter.data[i] / Scalar(2.0);
        Scalar ai_cube = ai*ai*ai;

        // Initialize Fi, Ei
        Scalar Fi_x = Scalar(0.0);
        Scalar Fi_y = Scalar(0.0);
        Scalar Fi_z = Scalar(0.0);
        Scalar Ei_x = Scalar(0.0);
        Scalar Ei_y = Scalar(0.0);
        Scalar Ei_z = Scalar(0.0);

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[idx];
        const unsigned int size = (unsigned int)h_n_neigh.data[idx];
        
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 rij = pi - pj;

            // apply periodic boundary conditions
            rij = box.minImage(rij);

            // add forces only if neighbor is within rcut distance
            Scalar rij_mag = slow::sqrt(dot(rij,rij));
            if (rij_mag < m_rcut)
                {
                Scalar rij_inv5 = Scalar(1.0) / (rij_mag * rij_mag * rij_mag * rij_mag * rij_mag);
                Scalar rij_inv2 = Scalar(1.0) / (rij_mag * rij_mag);

                Fi_x += rij_inv5 * rij.x * (3 - 15 * rij.z * rij.z * rij_inv2);
                Fi_y += rij_inv5 * rij.y * (3 - 15 * rij.z * rij.z * rij_inv2);
                Fi_z += rij_inv5 * rij.z * (6 + 3 - 15 * rij.z * rij.z * rij_inv2);

                Ei_x += 3 * rij_inv5 * rij.x * rij.z;
                Ei_y += 3 * rij_inv5 * rij.y * rij.z;
                Ei_z += 3 * rij_inv5 * rij.z * rij.z - rij_inv2/rij_mag;
                

                // add forces for the images
                Scalar3 rij_img;
                Scalar rij_mag_img;

                Scalar rjj_img_zplus = Scalar(0.0);
                Scalar rjj_img_zminus = Scalar(0.0);
                Scalar x = m_H / Scalar(2.0) - pi.z + rij.z ;
                Scalar sign = Scalar(1.0); // takes care of the inversion of reflected dipoles
                for (unsigned int zimage = 1; zimage < 1+m_rcut/m_H; zimage++)
                    {   
                        // add images along +z axis
                        sign *= -1;
                        rjj_img_zplus += -sign * Scalar(2.0)*x + m_H + sign * m_H;
                        rij_img = make_scalar3(rij.x, rij.y, rij.z - rjj_img_zplus);
                        rij_mag_img = slow::sqrt(dot(rij_img,rij_img));
                        if (rij_mag_img < m_rcut)
                            {
                            Scalar rij_inv5 = Scalar(1.0) / (rij_mag_img * rij_mag_img * rij_mag_img * rij_mag_img * rij_mag_img);
                            Scalar rij_inv2 = Scalar(1.0) / (rij_mag_img * rij_mag_img);

                            Fi_x += sign* rij_inv5 * rij_img.x * (3 - 15 * rij_img.z * rij_img.z * rij_inv2);
                            Fi_y += sign* rij_inv5 * rij_img.y * (3 - 15 * rij_img.z * rij_img.z * rij_inv2);
                            Fi_z += sign* rij_inv5 * rij_img.z * (6 + 3 - 15 * rij.z * rij_img.z * rij_inv2);
                            
                            Ei_x += sign* 3 * rij_inv5 * rij_img.x * rij_img.z;
                            Ei_y += sign* 3 * rij_inv5 * rij_img.y * rij_img.z;
                            Ei_z += sign* 3 * rij_inv5 * rij_img.z * rij_img.z - rij_inv2/rij_mag_img;
                            }
                    
                        // add images along -z axis
                        rjj_img_zminus += sign * Scalar(2.0)*x + m_H - sign * m_H;
                        rij_img = make_scalar3(rij.x, rij.y, rij.z + rjj_img_zminus);
                        rij_mag_img = slow::sqrt(dot(rij_img,rij_img));
                        if (rij_mag_img < m_rcut)
                            {
                            Scalar rij_inv5 = Scalar(1.0) / (rij_mag_img * rij_mag_img * rij_mag_img * rij_mag_img * rij_mag_img);
                            Scalar rij_inv2 = Scalar(1.0) / (rij_mag_img * rij_mag_img);

                            Fi_x += sign* rij_inv5 * rij_img.x * (3 - 15 * rij_img.z * rij_img.z * rij_inv2);
                            Fi_y += sign* rij_inv5 * rij_img.y * (3 - 15 * rij_img.z * rij_img.z * rij_inv2);
                            Fi_z += sign* rij_inv5 * rij_img.z * (6 + 3 - 15 * rij.z * rij_img.z * rij_inv2);
                            
                            Ei_x += sign* 3 * rij_inv5 * rij_img.x * rij_img.z;
                            Ei_y += sign* 3 * rij_inv5 * rij_img.y * rij_img.z;
                            Ei_z += sign* 3 * rij_inv5 * rij_img.z * rij_img.z - rij_inv2/rij_mag_img;
                            }
                    }
                }
            }
        
        //add images for i (i_iimage)
        Scalar3 rii_img;
        Scalar rii_mag_img;

        Scalar rii_img_zplus = Scalar(0.0);
        Scalar rii_img_zminus = Scalar(0.0);
        Scalar xii = m_H / Scalar(2.0) - pi.z ;
        Scalar sign = Scalar(1.0); // takes care of the inversion of reflected dipoles
        for (unsigned int zimage = 1; zimage < 1+m_rcut/m_H; zimage++)
            {   
                // add images along +z axis
                sign *= -1;
                rii_img_zplus += -sign * Scalar(2.0)*xii + m_H + sign * m_H;
                rii_img = make_scalar3(0, 0, -rii_img_zplus);
                rii_mag_img = slow::sqrt(dot(rii_img,rii_img));
                if (rii_mag_img < m_rcut)
                    {
                    Scalar rii_inv5 = Scalar(1.0) / (rii_mag_img * rii_mag_img * rii_mag_img * rii_mag_img * rii_mag_img);
                    Scalar rii_inv2 = Scalar(1.0) / (rii_mag_img * rii_mag_img);

                    Fi_x += sign* rii_inv5 * rii_img.x * (3 - 15 * rii_img.z * rii_img.z * rii_inv2);
                    Fi_y += sign* rii_inv5 * rii_img.y * (3 - 15 * rii_img.z * rii_img.z * rii_inv2);
                    Fi_z += sign* rii_inv5 * rii_img.z * (6 + 3 - 15 * rii.z * rii_img.z * rii_inv2);
                            
                    Ei_x += sign* 3 * rii_inv5 * rii_img.x * rii_img.z;
                    Ei_y += sign* 3 * rii_inv5 * rii_img.y * rii_img.z;
                    Ei_z += sign* 3 * rii_inv5 * rii_img.z * rii_img.z - rii_inv2/rii_mag_img;
                    }
                // add images along -z axis
                rii_img_zminus += sign * Scalar(2.0)*xii + m_H - sign * m_H;
                rii_img = make_scalar3(0, 0,  rii_img_zminus);
                rii_mag_img = slow::sqrt(dot(rii_img,rii_img));
                if (rii_mag_img < m_rcut)
                    {
                    Scalar rii_inv5 = Scalar(1.0) / (rii_mag_img * rii_mag_img * rii_mag_img * rii_mag_img * rii_mag_img);
                    Scalar rii_inv2 = Scalar(1.0) / (rii_mag_img * rii_mag_img);

                    Fi_x += sign* rii_inv5 * rii_img.x * (3 - 15 * rii_img.z * rii_img.z * rii_inv2);
                    Fi_y += sign* rii_inv5 * rii_img.y * (3 - 15 * rii_img.z * rii_img.z * rii_inv2);
                    Fi_z += sign* rii_inv5 * rii_img.z * (6 + 3 - 15 * rii.z * rii_img.z * rii_inv2);
                    
                    Ei_x += sign* 3 * rii_inv5 * rii_img.x * rii_img.z;
                    Ei_y += sign* 3 * rii_inv5 * rii_img.y * rii_img.z;
                    Ei_z += sign* 3 * rii_inv5 * rii_img.z * rii_img.z - rii_inv2/rii_mag_img;
                    }
            }


        // std::cout<< Fi_x << "**" << Fi_y << "**" << Fi_z << "**";

        Scalar Ei_mag = m_Ee * sqrt( (ai_cube * m_sigma21*Ei_x) * (ai_cube * m_sigma21*Ei_x) + 
                                     (ai_cube * m_sigma21*Ei_y) * (ai_cube * m_sigma21*Ei_y) + 
                                     (1 + ai_cube * m_sigma21*Ei_z) * (1 + ai_cube * m_sigma21*Ei_z));


        Scalar fnet_x = Fi_x * Scalar(4.0)*Scalar(3.1415926536) * m_epsilon * ai_cube * ai_cube * m_sigma21 * m_sigma21 * m_Ee * m_Ee ;
        Scalar fnet_y = Fi_y * Scalar(4.0)*Scalar(3.1415926536) * m_epsilon * ai_cube * ai_cube * m_sigma21 * m_sigma21 * m_Ee * m_Ee ;
        Scalar fnet_z = Fi_z * Scalar(4.0)*Scalar(3.1415926536) * m_epsilon * ai_cube * ai_cube * m_sigma21 * m_sigma21 * m_Ee * m_Ee ;


        Scalar D;
        if (Ei_mag > m_Ecut)
            D = m_Dactive;
        else
            D = m_Dpassive;

        // Introduce diffusion depending on Ei_mag is > m_Ecut
        // Initialize the RNG
        hoomd::RandomGenerator rng(0xfa9756f2, m_seed, tag, timestep);

        // compute the random force
        hoomd::UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
        Scalar rx = uniform(rng);
        Scalar ry = uniform(rng);
        Scalar rz = uniform(rng);

        Scalar gamma = Scalar(1.0); // Diffusion determines the quincke force
        
        // compute the brownian force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1 distribution
        // it is not the dimensionality of the system
        Scalar coeff = gamma * fast::sqrt(Scalar(3.0)*Scalar(2.0)*D/m_deltaT);
        
        fnet_x += rx*coeff;
        fnet_y += ry*coeff;
        fnet_z += rz*coeff;


        h_force.data[idx].x = fnet_x;
        h_force.data[idx].y = fnet_y;
        h_force.data[idx].z = fnet_z;


        }
    
    }


/*! This function applies constraints, rotational diffusion, and sets forces for all active particles
    \param timestep Current timestep
*/
void QuinckeForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push(m_exec_conf, "QuinckeForceCompute");

    if (last_computed != timestep)
        {
        //m_rotationConst = slow::sqrt(2.0 * m_rotationDiff * m_deltaT);

        last_computed = timestep;

        // if (m_rx != 0)
        //     {
        //     setConstraint(); // apply surface constraints to active particles active force vectors
        //     }
        // if (m_rotationDiff != 0)
        //     {
        //     rotationalDiffusion(timestep); // apply rotational diffusion to active particles
        //     }
        setForces(timestep); // set forces for particles
        }

    #ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    #endif

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


/* Export the CPU updater to be visible in the python module
 */
void export_QuinckeForceCompute(pybind11::module& m)
    {
    pybind11::class_<QuinckeForceCompute, std::shared_ptr<QuinckeForceCompute> >(m, "QuinckeForceCompute", pybind11::base<ForceCompute>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, std::shared_ptr<ParticleGroup>, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, int >())
    ;
    }

// ********************************
// For future: here follows the code for QuinckeForceCompute on the GPU

