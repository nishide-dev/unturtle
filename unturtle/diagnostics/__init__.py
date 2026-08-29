# Copyright 2025-present nishide-dev & the Unturtle team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observation-only diagnostics (#184).

Nothing in this package may be imported by production loaders, trainers or
generation code — it exists so the architecture-characterization producer
(``benchmarks/architecture/``) and its tests can observe runtime contracts
without adding a production dependency. The dependency direction is enforced
by ``tests/architecture/test_contract_artifact.py``.
"""
