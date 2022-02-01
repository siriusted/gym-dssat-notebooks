#!/bin/bash
# install system dependencies
echo "Starting installation procedure"
echo "Installing system dependencies..."
apt-get update &> /dev/null
apt-get install python3.7-dev python3.7-venv &> /dev/null
wget http://pyyaml.org/download/libyaml/yaml-0.2.5.tar.gz &> /dev/null
tar -xf yaml-0.2.5.tar.gz &> /dev/null
cd yaml-0.2.5 && ./configure &> /dev/null && make &> /dev/null && make install &> /dev/null
cd ../

# install pip dependencies 
# only need gym==0.18.3, pyyaml > 5.1
pip install gym==0.18.3 &> /dev/null
pip install -U PyYAML &> /dev/null
echo "Done"
echo
echo "Installing pdi..."
# install pdi
git clone https://gitlab.maisondelasimulation.fr/pdidev/pdi.git &> /dev/null
mkdir pdi/build && cd pdi/build
cmake -DCMAKE_INSTALL_PREFIX='/opt/pdi' \
    -DDIST_PROFILE=User \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DBUILD_CFG_VALIDATOR=OFF \
    -DBUILD_DECL_HDF5_PLUGIN=OFF \
    -DBUILD_DECL_NETCDF_PLUGIN=OFF \
    -DBUILD_DECL_SION_PLUGIN=OFF \
    -DBUILD_FLOWVR_PLUGIN=OFF \
    -DBUILD_FORTRAN=ON \
    -DBUILD_FTI_PLUGIN=OFF \
    -DBUILD_HDF5_PARALLEL=OFF \
    -DBUILD_MPI_PLUGIN=OFF \
    -DBUILD_PYCALL_PLUGIN=ON \
    -DBUILD_PYTHON=ON \
    -DBUILD_SET_VALUE_PLUGIN=ON \
    -DBUILD_SERIALIZE_PLUGIN=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TEST_PLUGIN=OFF \
    -DBUILD_TRACE_PLUGIN=ON \
    -DBUILD_USER_CODE_PLUGIN=ON \
    -DUSE_DEFAULT=EMBEDDED .. &> /dev/null
make install &> /dev/null
echo "Done"
echo
echo "Installing dssat-pdi..."

# install dssat-pdi
cd ../../
git clone --recurse-submodules https://gitlab.inria.fr/rgautron/gym_dssat_pdi.git &> /dev/null
cd gym_dssat_pdi/dssat-csm-os && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX='/opt/dssat_pdi' -DUSE_DEFAULT=EMBEDDED -DCMAKE_PREFIX_PATH='/opt/pdi/share/paraconf/cmake;/opt/pdi/share/pdi/cmake' .. &> /dev/null
make &> /dev/null && make install &> /dev/null
cd ../../dssat-csm-data && cp -r ./* /opt/dssat_pdi
echo "Done"
echo
echo "Installing gym-dssat..."


# install gym-dssat
cd ../gym-dssat-pdi/ && pip install -e . &> /dev/null
echo "Done"
echo

echo "Installation procedure completed!"
