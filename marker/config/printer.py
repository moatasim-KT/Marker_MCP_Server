from typing import Optional

import click

from marker.config.crawler import crawler


class CustomClickPrinter(click.Command):
    def parse_args(self, ctx, args):
        display_help = "config" in args and "--help" in args
        if display_help:
            click.echo(
                "Here is a list of all the Builders, Processors, Converters, Providers and Renderers in Marker along with their attributes:"
            )

        # Keep track of shared attributes and their types
        shared_attrs = {}

        # First pass: identify shared attributes and verify compatibility
        for base_type, base_type_dict in crawler.class_config_map.items():
            for class_name, class_map in base_type_dict.items():
                for attr, (attr_type, formatted_type, default, metadata) in class_map[
                    "config"
                ].items():
                    if attr not in shared_attrs:
                        shared_attrs[attr] = {
                            "classes": [],
                            "type": attr_type,
                            "is_flag": attr_type in [bool, Optional[bool]]
                            and not default,
                            "metadata": metadata,
                            "default": default,
                        }
                    shared_attrs[attr]["classes"].append(class_name)

        # These are the types of attrs that can be set from the command line
        attr_types = [
            str,
            int,
            float,
            bool,
            Optional[int],
            Optional[float],
            Optional[str],
        ]

        # Track added option names to avoid duplicates (Click treats --foo and --foo/--bar as different)
        added_options = set()
        # Skip attributes already handled by ConfigParser to avoid duplicates
        skip_attrs = {"extract_images", "disable_multiprocessing", "groq_api_key", "groq_model_name", "groq_base_url"}
        # Add shared attribute options first
        for attr, info in shared_attrs.items():
            if attr in skip_attrs:
                continue
            if info["type"] in attr_types:
                option_name = "--" + attr
                if option_name in added_options:
                    continue
                # Only add as is_flag for booleans, never as type=bool
                if info["is_flag"]:
                    ctx.command.params.append(
                        click.Option(
                            [option_name],
                            is_flag=True,
                            help=" ".join(info["metadata"]) + f" (Applies to: {', '.join(info['classes'])})",
                        )
                    )
                else:
                    ctx.command.params.append(
                        click.Option(
                            [option_name],
                            type=info["type"],
                            help=" ".join(info["metadata"]) + f" (Applies to: {', '.join(info['classes'])})",
                            default=None,
                        )
                    )
                added_options.add(option_name)
        # Second pass: create class-specific options
        for base_type, base_type_dict in crawler.class_config_map.items():
            if display_help:
                click.echo(f"{base_type}s:")
            for class_name, class_map in base_type_dict.items():
                if display_help and class_map["config"]:
                    click.echo(
                        f"\n  {class_name}: {class_map['class_type'].__doc__ or ''}"
                    )
                    click.echo(" " * 4 + "Attributes:")
                for attr, (attr_type, formatted_type, default, metadata) in class_map[
                    "config"
                ].items():
                    class_name_attr = class_name + "_" + attr
                    option_name = "--" + class_name_attr
                    if option_name in added_options:
                        continue
                    if attr_type in attr_types:
                        is_flag = attr_type in [bool, Optional[bool]] and not default
                        # Only add as is_flag for booleans, never as type=bool
                        if is_flag:
                            ctx.command.params.append(
                                click.Option(
                                    [option_name, class_name_attr],
                                    is_flag=True,
                                    help=" ".join(metadata),
                                )
                            )
                        else:
                            ctx.command.params.append(
                                click.Option(
                                    [option_name, class_name_attr],
                                    type=attr_type,
                                    help=" ".join(metadata),
                                    default=None,
                                )
                            )
                        added_options.add(option_name)

        if display_help:
            ctx.exit()

        super().parse_args(ctx, args)
