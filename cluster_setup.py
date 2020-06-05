from dask.distributed import Client, LocalCluster
import os
def setup_cluster(run_strategy = "local", ncore=2, nnodes = 1):
    if run_strategy=="local":
        cluster = LocalCluster(n_workers=ncore, threads_per_worker=1)
    elif run_strategy == "PBSjobqueue":
        from dask_jobqueue import PBSCluster
        cluster = PBSCluster(cores=ncore,
            processes=ncore,
            resource_spec=f"nodes=1:ppn={ncore}",
            group='wagner',
            queue='secondary',
            memory='16G',
            walltime='02:00:00',
            env_extra=['cd ${PBS_O_WORKDIR}',
                            'export PYTHONPATH=/home/lkwagner/pyqmc:$PYTHONPATH',
                            'export OMP_NUM_THREADS=1',
                            'source /home/lkwagner/.bashrc',
                            'conda activate pyscf'],
                    local_directory=os.getenv('TMPDIR', '/tmp'))
        cluster.submit_command="/usr/local/torque-releases/torque-6.1.2-el7/bin/qsub"
        cluster.cancel_command="/usr/local/torque-releases/torque-6.1.2-e17/bin/qdel"
        print(cluster.job_script())
        cluster.scale(nnodes)

    return Client(cluster), cluster
