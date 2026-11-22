plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// The deployed model and MediaPipe assets live in ../../models (single source of truth);
// they are copied into the APK at build time.
val modelAssets = layout.buildDirectory.dir("generated/modelAssets")
val copyModelAssets by tasks.registering(Copy::class) {
    from(rootProject.file("../models")) {
        include("echogesture.int8.onnx", "echogesture.int8.json", "hand_landmarker.task", "pose_landmarker_lite.task")
    }
    into(modelAssets)
}

android {
    namespace = "com.echogesture.edge"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.echogesture.edge"
        minSdk = 24          // Android 7.0+
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            // Every Android phone is ARM: 64-bit (arm64-v8a) or older 32-bit (armeabi-v7a).
            // MediaPipe ships no x86 libraries, so x86 emulators are not supported.
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    signingConfigs {
        // A fixed, committed key so every build installs over the previous one on any phone.
        // For sideloading/demo only: use a private key for a store release.
        create("demo") {
            storeFile = file("echogesture-demo.jks")
            storePassword = "echogesture"
            keyAlias = "echogesture"
            keyPassword = "echogesture"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("demo")
        }
        debug {
            signingConfig = signingConfigs.getByName("demo")
        }
    }

    sourceSets["main"].assets.srcDir(modelAssets)

    androidResources {
        noCompress += listOf("onnx", "task")
    }

    packaging {
        // Compressed native libs: much smaller APK to share/download; extracted at install.
        jniLibs.useLegacyPackaging = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

tasks.named("preBuild") { dependsOn(copyModelAssets) }

dependencies {
    val camerax = "1.4.2"
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    // Same MediaPipe 0.10 Tasks API and the same bundled .task models as the Python extractor
    // (0.10.21); 0.10.26.1 is the first release here whose arm64 library is 16 KB page-aligned,
    // which Android 15+ devices with 16 KB pages require.
    implementation("com.google.mediapipe:tasks-vision:0.10.26.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
