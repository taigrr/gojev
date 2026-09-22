// Package version exposes the gojev build version.
package version

import "runtime/debug"

// Version is the gojev version. It defaults to the module version embedded by
// the Go toolchain and can be overridden at build time via
// -ldflags "-X github.com/taigrr/gojev/internal/version.Version=...".
var Version = "devel"

func init() {
	if Version != "devel" {
		return
	}
	if info, ok := debug.ReadBuildInfo(); ok {
		if v := info.Main.Version; v != "" && v != "(devel)" {
			Version = v
		}
	}
}
