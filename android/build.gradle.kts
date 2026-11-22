plugins {
    id("com.android.application") version "8.10.1" apply false
    id("org.jetbrains.kotlin.android") version "2.1.21" apply false
}

// Optional: build outside the source tree, e.g. when the repo lives on an exFAT/FAT drive where
// macOS writes "._*" AppleDouble files that break resource packaging:
//   ./gradlew assembleRelease -PbuildRoot=$HOME/.cache/echogesture-android
providers.gradleProperty("buildRoot").orNull?.let { root ->
    allprojects { layout.buildDirectory.set(file("$root/${project.name}")) }
}
