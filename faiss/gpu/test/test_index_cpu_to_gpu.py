import numpy as np
import unittest
import faiss


class TestMoveToGpu(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.res = faiss.StandardGpuResources()

    def create_index(self, factory_string):
        dimension = 128
        n = 2500
        db_vectors = np.random.random((n, dimension)).astype('float32')
        index = faiss.index_factory(dimension, factory_string)
        index.train(db_vectors)
        if factory_string.startswith("IDMap"):
            index.add_with_ids(db_vectors, np.arange(n))
        else:
            index.add(db_vectors)
        return index

    def create_and_clone(self, factory_string,
                         enableCpuFallback=None,
                         use_raft=None):
        idx = self.create_index(factory_string)
        config = faiss.GpuClonerOptions()
        if enableCpuFallback is not None:
            config.enableCpuFallback = enableCpuFallback
        if use_raft is not None:
            config.use_raft = use_raft
        faiss.index_cpu_to_gpu(self.res, 0, idx, config)

    def verify_throws_on_unsupported_index(self, factory_string):
        try:
            self.create_and_clone(factory_string)
        except Exception as e:
            if "not implemented" not in str(e):
                self.fail("Expected an exception but no exception was "
                          "thrown for factory_string: %s." % factory_string)

    def verify_succeeds_on_supported_index(self, factory_string, use_raft=None):
        try:
            self.create_and_clone(factory_string, use_raft=use_raft)
        except Exception as e:
            self.fail("Unexpected exception thrown factory_string: "
                      "%s; error message: %s." % (factory_string, str(e)))

    def verify_succeeds_on_unsupported_index_with_fallback_enabled(
            self, factory_string, use_raft=None):
        try:
            self.create_and_clone(factory_string, enableCpuFallback=True,
                                  use_raft=use_raft)
        except Exception as e:
            self.fail("Unexpected exception thrown factory_string: "
                      "%s; error message: %s." % (factory_string, str(e)))

    def test_index_cpu_to_gpu_unsupported_indices(self):
        self.verify_throws_on_unsupported_index("PQ16")
        self.verify_throws_on_unsupported_index("LSHrt")
        self.verify_throws_on_unsupported_index("HNSW")
        self.verify_throws_on_unsupported_index("HNSW,PQ16")
        self.verify_throws_on_unsupported_index("IDMap,PQ16")
        self.verify_throws_on_unsupported_index("IVF256,ITQ64,SH1.2")

    def test_index_cpu_to_gpu_supported_indices(self):
        self.verify_succeeds_on_supported_index("Flat")
        self.verify_succeeds_on_supported_index("IVF1,Flat")
        self.verify_succeeds_on_supported_index("IVF32,PQ8")

        # set use_raft to false, this index type is not supported on RAFT
        self.verify_succeeds_on_supported_index("IVF32,SQ8", use_raft=False)

    def test_index_cpu_to_gpu_unsupported_indices_with_fallback_enabled(self):
        self.verify_succeeds_on_unsupported_index_with_fallback_enabled("IDMap,Flat")
        self.verify_succeeds_on_unsupported_index_with_fallback_enabled("PCA12,IVF32,Flat")
        self.verify_succeeds_on_unsupported_index_with_fallback_enabled("PCA32,IVF32,PQ8")
        self.verify_succeeds_on_unsupported_index_with_fallback_enabled("PCA32,IVF32,PQ8np")

        # set use_raft to false, this index type is not supported on RAFT
        self.verify_succeeds_on_unsupported_index_with_fallback_enabled(
            "PCA32,IVF32,SQ8", use_raft=False)
