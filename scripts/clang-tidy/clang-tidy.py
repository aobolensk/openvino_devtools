#!/usr/bin/env python3
"""
OpenVINO Clang-Tidy Multi-Architecture Build Script

This script runs clang-tidy enabled builds for multiple architectures:
- x64 (native)
- arm64 (cross-compile)
- riscv64 (cross-compile)

The script uses Docker to ensure consistent build environment and supports
parallel builds with ccache for improved performance.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ClangTidyBuilder:
    """Manages clang-tidy builds for multiple architectures."""
    
    ARCHITECTURES = {
        'x64': {
            'cmake_generator': 'Ninja Multi-Config',
            'cmake_args': [],
            'toolchain': None,
            'build_args': ['--quiet', '-k', '0']
        },
        'arm64': {
            'cmake_generator': 'Unix Makefiles',
            'cmake_args': ['-DCMAKE_RULE_MESSAGES=OFF'],
            'toolchain': None,  # Uses system cross-compiler
            'build_args': ['-k']
        },
        'riscv64': {
            'cmake_generator': 'Unix Makefiles',
            'cmake_args': ['-DCMAKE_RULE_MESSAGES=OFF'],
            'toolchain': 'cmake/toolchains/riscv64-100-xuantie-gnu.toolchain.cmake',
            'build_args': ['-k']
        }
    }
    
    DEFAULT_DOCKER_IMAGE = 'ghcr.io/aobolensk/openvino_devutils/ubuntu:latest'
    
    def __init__(self, 
                 openvino_repo: str,
                 build_root: str = None,
                 docker_image: str = None,
                 architectures: List[str] = None,
                 target: str = 'openvino_intel_cpu_plugin',
                 jobs: int = None,
                 verbose: bool = False):
        """
        Initialize the builder.
        
        Args:
            openvino_repo: Path to OpenVINO repository
            build_root: Root directory for build outputs (default: temp dir)
            docker_image: Docker image to use for builds
            architectures: List of architectures to build (default: all)
            target: CMake target to build
            jobs: Number of parallel jobs (default: nproc - 1)
            verbose: Enable verbose logging
        """
        self.openvino_repo = Path(openvino_repo).resolve()
        self.build_root = Path(build_root) if build_root else Path.cwd() / 'build'
        self.docker_image = docker_image or self.DEFAULT_DOCKER_IMAGE
        self.architectures = architectures or list(self.ARCHITECTURES.keys())
        self.target = target
        self.jobs = jobs
        self.verbose = verbose
        
        # Validate inputs
        if not self.openvino_repo.exists():
            raise ValueError(f"OpenVINO repository not found: {self.openvino_repo}")
        
        invalid_archs = set(self.architectures) - set(self.ARCHITECTURES.keys())
        if invalid_archs:
            raise ValueError(f"Invalid architectures: {invalid_archs}")
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create build directories
        self.build_root.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"OpenVINO repository: {self.openvino_repo}")
        self.logger.info(f"Build root: {self.build_root}")
        self.logger.info(f"Docker image: {self.docker_image}")
        self.logger.info(f"Architectures: {', '.join(self.architectures)}")
        self.logger.info(f"Target: {self.target}")

    def _run_command(self, cmd: List[str], cwd: Path = None, env: Dict[str, str] = None) -> Tuple[int, str, str]:
        """
        Run a command and return exit code, stdout, stderr.
        
        Args:
            cmd: Command and arguments
            cwd: Working directory
            env: Environment variables
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        self.logger.debug(f"Running: {' '.join(cmd)}")
        if cwd:
            self.logger.debug(f"Working directory: {cwd}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if self.verbose and result.stdout:
                self.logger.debug(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                level = logging.DEBUG if result.returncode == 0 else logging.WARNING
                self.logger.log(level, f"STDERR:\n{result.stderr}")
                
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.error("Command timed out after 2 hours")
            return 124, "", "Command timed out"
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            return 1, "", str(e)

    def _check_docker(self) -> bool:
        """Check if Docker is available and build/pull image as needed."""
        self.logger.info("Checking Docker availability...")
        
        # Check if docker command exists
        exit_code, _, _ = self._run_command(['docker', '--version'])
        if exit_code != 0:
            self.logger.error("Docker is not available or not installed")
            return False
        
        # Check if image exists locally
        exit_code, _, _ = self._run_command(['docker', 'inspect', self.docker_image])
        if exit_code == 0:
            self.logger.info(f"Docker image {self.docker_image} found locally")
            return True
        
        # Try to pull the image first
        self.logger.info(f"Pulling Docker image: {self.docker_image}")
        exit_code, _, stderr = self._run_command(['docker', 'pull', self.docker_image])
        if exit_code == 0:
            return True
        
        # If pull failed, try to build locally from Dockerfile
        self.logger.warning(f"Failed to pull image: {stderr}")
        return self._build_docker_image()
    
    def _build_docker_image(self) -> bool:
        """Build Docker image locally from the provided Dockerfile."""
        dockerfile_path = Path(__file__).parent.parent.parent / 'docker' / 'ubuntu.Dockerfile'
        
        if not dockerfile_path.exists():
            self.logger.error(f"Dockerfile not found: {dockerfile_path}")
            return False
        
        self.logger.info(f"Building Docker image locally from {dockerfile_path}")
        
        # Determine the correct sccache architecture
        import platform
        machine = platform.machine()
        if machine == 'arm64' or machine == 'aarch64':
            sccache_arch = 'aarch64-unknown-linux-musl'
        else:
            sccache_arch = 'x86_64-unknown-linux-musl'
        
        build_cmd = [
            'docker', 'build',
            '--build-arg', f'SCCACHE_ARCH={sccache_arch}',
            '-f', str(dockerfile_path),
            '-t', self.docker_image,
            str(dockerfile_path.parent)
        ]
        
        exit_code, stdout, stderr = self._run_command(build_cmd)
        
        if exit_code != 0:
            self.logger.error(f"Failed to build Docker image: {stderr}")
            return False
        
        self.logger.info("Docker image built successfully")
        return True

    def _get_docker_run_cmd(self, arch: str, mount_paths: Dict[str, str]) -> List[str]:
        """
        Generate docker run command for the given architecture.
        
        Args:
            arch: Target architecture
            mount_paths: Dictionary of host_path -> container_path mappings
            
        Returns:
            Docker run command as list
        """
        cmd = [
            'docker', 'run', '--rm', '-i',
            '--workdir', '/workspace',
        ]
        
        # Add volume mounts
        for host_path, container_path in mount_paths.items():
            cmd.extend(['-v', f'{host_path}:{container_path}'])
        
        # Set environment variables
        env_vars = {
            'DEBIAN_FRONTEND': 'noninteractive',
            'CMAKE_BUILD_TYPE': 'Release',
            'CMAKE_CXX_COMPILER_LAUNCHER': 'ccache',
            'CMAKE_C_COMPILER_LAUNCHER': 'ccache',
            'CMAKE_COMPILE_WARNING_AS_ERROR': 'ON',
            'CCACHE_DIR': '/workspace/.ccache',
            'CCACHE_MAXSIZE': '10G',
        }
        
        if arch == 'riscv64':
            env_vars['RISCV_TOOLCHAIN_ROOT'] = '/opt/riscv'
        
        for key, value in env_vars.items():
            cmd.extend(['-e', f'{key}={value}'])
        
        cmd.append(self.docker_image)
        
        return cmd

    def _configure_cmake(self, arch: str, build_dir: Path, source_dir: Path) -> bool:
        """
        Configure CMake for the given architecture.
        
        Args:
            arch: Target architecture
            build_dir: Build directory
            source_dir: Source directory
            
        Returns:
            True if configuration succeeded
        """
        self.logger.info(f"Configuring CMake for {arch}...")
        
        arch_config = self.ARCHITECTURES[arch]
        
        cmake_args = [
            'cmake',
            '-G', arch_config['cmake_generator'],
            '-DENABLE_CLANG_TIDY=ON',
            '-DENABLE_PROFILING_ITT=ON',
            '-DENABLE_DEBUG_CAPS=ON',
            '-DSELECTIVE_BUILD=COLLECT',
            '-DENABLE_PYTHON=OFF',
            '-DENABLE_TESTS=OFF',
            '-DENABLE_NCC_STYLE=OFF',
            '-DENABLE_CPPLINT=OFF',
            '-DENABLE_FASTER_BUILD=OFF',
            '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
            '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
        ]
        
        # Add architecture-specific CMake arguments
        cmake_args.extend(arch_config['cmake_args'])
        
        # Add toolchain file if specified
        if arch_config['toolchain']:
            cmake_args.extend(['-DCMAKE_TOOLCHAIN_FILE', arch_config['toolchain']])
        
        # Add source and build directories
        cmake_args.extend(['-S', str(source_dir), '-B', str(build_dir)])
        
        # Prepare Docker command
        mount_paths = {
            str(self.openvino_repo): '/workspace/openvino',
            str(self.build_root.resolve()): '/workspace/build',
        }
        
        docker_cmd = self._get_docker_run_cmd(arch, mount_paths)
        # Use bash -c to wrap the cmake command
        cmake_cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmake_args)
        docker_cmd.extend(['bash', '-c', cmake_cmd_str])
        
        # Replace paths in cmake args to use container paths
        for i, arg in enumerate(docker_cmd):
            if arg == str(source_dir):
                docker_cmd[i] = '/workspace/openvino'
            elif arg == str(build_dir):
                docker_cmd[i] = f'/workspace/build/build_{arch}'
        
        exit_code, stdout, stderr = self._run_command(docker_cmd)
        
        if exit_code != 0:
            self.logger.error(f"CMake configuration failed for {arch}")
            self.logger.error(f"Exit code: {exit_code}")
            if stderr:
                self.logger.error(f"Error output:\n{stderr}")
            return False
        
        self.logger.info(f"CMake configuration completed for {arch}")
        return True

    def _build_target(self, arch: str, build_dir: Path) -> bool:
        """
        Build the target for the given architecture.
        
        Args:
            arch: Target architecture
            build_dir: Build directory
            
        Returns:
            True if build succeeded
        """
        self.logger.info(f"Building {self.target} for {arch}...")
        
        arch_config = self.ARCHITECTURES[arch]
        
        # Determine number of parallel jobs
        if self.jobs:
            parallel_jobs = self.jobs
        else:
            # Use nproc - 1, but at least 1
            try:
                nproc = int(subprocess.check_output(['nproc'], text=True).strip())
                parallel_jobs = max(1, nproc - 1)
            except:
                parallel_jobs = 4  # fallback
        
        cmake_build_args = [
            'cmake',
            '--build', str(build_dir),
            '--parallel', str(parallel_jobs),
            '--config', 'Release',
            '--target', self.target,
        ]
        
        # Add architecture-specific build arguments
        if arch_config['build_args']:
            cmake_build_args.extend(['--'] + arch_config['build_args'])
        
        # Prepare Docker command
        mount_paths = {
            str(self.openvino_repo): '/workspace/openvino',
            str(self.build_root.resolve()): '/workspace/build',
        }
        
        docker_cmd = self._get_docker_run_cmd(arch, mount_paths)
        # Use bash -c to wrap the cmake build command
        cmake_cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmake_build_args)
        docker_cmd.extend(['bash', '-c', cmake_cmd_str])
        
        # Replace build directory path to use container path
        for i, arg in enumerate(docker_cmd):
            if arg == str(build_dir):
                docker_cmd[i] = f'/workspace/build/build_{arch}'
        
        exit_code, stdout, stderr = self._run_command(docker_cmd)
        
        if exit_code != 0:
            self.logger.error(f"Build failed for {arch}")
            self.logger.error(f"Exit code: {exit_code}")
            if stderr:
                self.logger.error(f"Error output:\n{stderr}")
            return False
        
        self.logger.info(f"Build completed successfully for {arch}")
        return True

    def _show_ccache_stats(self, arch: str) -> None:
        """Show ccache statistics for the given architecture."""
        self.logger.info(f"Showing ccache stats for {arch}...")
        
        mount_paths = {
            str(self.build_root.resolve()): '/workspace/build',
        }
        
        docker_cmd = self._get_docker_run_cmd(arch, mount_paths)
        docker_cmd.extend(['bash', '-c', 'ccache --show-stats'])
        
        exit_code, stdout, stderr = self._run_command(docker_cmd)
        
        if exit_code == 0 and stdout:
            self.logger.info(f"Ccache stats for {arch}:\n{stdout}")
        else:
            self.logger.warning(f"Failed to get ccache stats for {arch}")

    def build_architecture(self, arch: str) -> bool:
        """
        Build for a single architecture.
        
        Args:
            arch: Architecture to build
            
        Returns:
            True if build succeeded
        """
        self.logger.info(f"Starting build for architecture: {arch}")
        
        build_dir = self.build_root / f"build_{arch}"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure CMake
        if not self._configure_cmake(arch, build_dir, self.openvino_repo):
            return False
        
        # Build target
        if not self._build_target(arch, build_dir):
            return False
        
        # Show ccache stats
        self._show_ccache_stats(arch)
        
        self.logger.info(f"Successfully completed build for {arch}")
        return True

    def build_all(self) -> bool:
        """
        Build for all configured architectures.
        
        Returns:
            True if all builds succeeded
        """
        if not self._check_docker():
            return False
        
        success_count = 0
        failed_architectures = []
        
        for arch in self.architectures:
            try:
                if self.build_architecture(arch):
                    success_count += 1
                else:
                    failed_architectures.append(arch)
            except Exception as e:
                self.logger.error(f"Unexpected error building {arch}: {e}")
                failed_architectures.append(arch)
        
        # Summary
        total_archs = len(self.architectures)
        self.logger.info(f"Build summary: {success_count}/{total_archs} architectures succeeded")
        
        if failed_architectures:
            self.logger.error(f"Failed architectures: {', '.join(failed_architectures)}")
            return False
        
        self.logger.info("All builds completed successfully!")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run clang-tidy enabled builds for multiple architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build all architectures
  %(prog)s /path/to/openvino

  # Build only x64 and arm64
  %(prog)s /path/to/openvino --arch x64 arm64

  # Use custom Docker image and build directory
  %(prog)s /path/to/openvino --docker-image my/custom:image --build-dir ./builds

  # Build with verbose output and custom target
  %(prog)s /path/to/openvino --verbose --target openvino_intel_cpu_plugin
        """
    )
    
    parser.add_argument(
        'openvino_repo',
        help='Path to OpenVINO repository'
    )
    
    parser.add_argument(
        '--arch', '--architecture',
        dest='architectures',
        choices=['x64', 'arm64', 'riscv64'],
        nargs='+',
        default=['x64', 'arm64', 'riscv64'],
        help='Architectures to build (default: all)'
    )
    
    parser.add_argument(
        '--build-dir',
        dest='build_dir',
        help='Root directory for build outputs (default: ./build)'
    )
    
    parser.add_argument(
        '--docker-image',
        dest='docker_image',
        default=ClangTidyBuilder.DEFAULT_DOCKER_IMAGE,
        help=f'Docker image to use (default: {ClangTidyBuilder.DEFAULT_DOCKER_IMAGE})'
    )
    
    parser.add_argument(
        '--target',
        default='openvino_intel_cpu_plugin',
        help='CMake target to build (default: openvino_intel_cpu_plugin)'
    )
    
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        help='Number of parallel jobs (default: nproc - 1)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        builder = ClangTidyBuilder(
            openvino_repo=args.openvino_repo,
            build_root=args.build_dir,
            docker_image=args.docker_image,
            architectures=args.architectures,
            target=args.target,
            jobs=args.jobs,
            verbose=args.verbose
        )
        
        success = builder.build_all()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nBuild interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()