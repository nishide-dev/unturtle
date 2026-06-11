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

import typer

from unturtle.cli.commands.eval import eval_cmd
from unturtle.cli.commands.export import export, list_checkpoints
from unturtle.cli.commands.generate import generate
from unturtle.cli.commands.train import train

app = typer.Typer(
    help="Unturtle CLI — dLLM training, generation, and evaluation.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)

app.command()(train)
app.command()(generate)
app.command()(export)
app.command("list-checkpoints")(list_checkpoints)
app.command("eval")(eval_cmd)
