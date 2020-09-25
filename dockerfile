FROM nvidia/cuda:10.1-devel-ubuntu18.04
MAINTAINER Nifelabs, info@gonife.com

ARG DEBIAN_FRONTEND=noninteractive


# Install build and runtime dependencies, Intel OpenVINO and OpenCL drivers
RUN apt-get update && apt-get install -y curl
RUN echo "deb http://ppa.launchpad.net/intel-opencl/intel-opencl/ubuntu bionic main" >> /etc/apt/sources.list \
 && echo "deb https://apt.repos.intel.com/openvino/2019/ all main" >> /etc/apt/sources.list \
 && curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | apt-key add - \
 && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B9732172C4830B8F \
 && apt-get install -y --no-install-recommends python3-pip python3-dev  \
 && apt-get install -y  cpio \
 && apt-get update && apt-get install -y \
    build-essential \
    clinfo \
    intel-opencl-icd \
    intel-openvino-dev-ubuntu18-2019.3.344 \
    libgtk-3-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ocl-icd-libopencl1 \
    python3-pyqt5 \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG package_url=http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_2020.2.120.tgz
ARG TEMP_DIR=/tmp/openvino_installer
WORKDIR ${TEMP_DIR}
ADD ${package_url} ${TEMP_DIR}
# install product by installation script
ENV INTEL_OPENVINO_DIR /opt/intel/openvino
RUN tar -xzf ${TEMP_DIR}/*.tgz --strip 1
RUN sed -i 's/decline/accept/g' silent.cfg && \
    ${TEMP_DIR}/install.sh -s silent.cfg && \
    ${INTEL_OPENVINO_DIR}/install_dependencies/install_openvino_dependencies.sh
WORKDIR /tmp
RUN rm -rf ${TEMP_DIR}
# installing dependencies for package
WORKDIR /tmp

RUN python3 -m pip install --no-cache-dir setuptools && \
    find "${INTEL_OPENVINO_DIR}/" -type f -name "*requirements*.*" -path "*/python3/*" -exec python3 -m pip install --no-cache-dir -r "{}" \; && \
    find "${INTEL_OPENVINO_DIR}/" -type f -name "*requirements*.*" -not -path "*/post_training_optimization_toolkit/*" -not -name "*windows.txt"  -not -name "*ubuntu16.txt" -not -path "*/python3*/*" -not -path "*/python2*/*" -exec python3 -m pip install --no-cache-dir -r "{}" \;
WORKDIR ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checker
RUN source ${INTEL_OPENVINO_DIR}/bin/setupvars.sh && \
    python3 -m pip install --no-cache-dir -r ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checker/requirements.in && \
    python3 ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checker/setup.py install
WORKDIR ${INTEL_OPENVINO_DIR}/deployment_tools/tools/post_training_optimization_toolkit
RUN if [ -f requirements.txt ]; then \
        python3 -m pip install --no-cache-dir -r ${INTEL_OPENVINO_DIR}/deployment_tools/tools/post_training_optimization_toolkit/requirements.txt && \
        python3 ${INTEL_OPENVINO_DIR}/deployment_tools/tools/post_training_optimization_toolkit/setup.py install; \
    fi;
# Post-installation cleanup and setting up OpenVINO environment variables
RUN if [ -f "${INTEL_OPENVINO_DIR}"/bin/setupvars.sh ]; then \
        printf "\nsource \${INTEL_OPENVINO_DIR}/bin/setupvars.sh\n" >> /home/openvino/.bashrc; \
        printf "\nsource \${INTEL_OPENVINO_DIR}/bin/setupvars.sh\n" >> /root/.bashrc; \
    fi;
RUN find "${INTEL_OPENVINO_DIR}/" -name "*.*sh" -type f -exec dos2unix {} \;

# Prevent NVIDIA libOpenCL.so from being loaded
RUN mv /usr/local/cuda-10.1/targets/x86_64-linux/lib/libOpenCL.so.1 \
       /usr/local/cuda-10.1/targets/x86_64-linux/lib/libOpenCL.so.1.bak

# You can speed up build slightly by reducing build context with
#     git archive --format=tgz HEAD | docker build -t openrtist -
COPY . shudhOpenVino
WORKDIR shudhOpenVino

# Install PyTorch and Gabriel's external dependencies
#COPY python-client/requirements.txt client-requirements.txt
#COPY server/requirements.txt server-requirements.txt
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    -r requirements.txt 
#    -r server-requirements.txt

RUN chmod +x ./server.py
EXPOSE 5555 9099
ENTRYPOINT ["python3", "./server.py"]
