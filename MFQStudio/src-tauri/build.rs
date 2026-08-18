use icns::{IconFamily, Image as IcnsImage, PixelFormat};
use ico::{IconDir, IconDirEntry, IconImage, ResourceType};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

fn inside_background(x: f64, y: f64, inset: f64) -> bool {
    let half = 20.0 - inset;
    let radius = 9.375 - inset * 0.5;
    let qx = (x - 20.0).abs() - (half - radius);
    let qy = (y - 20.0).abs() - (half - radius);
    let outside = qx.max(0.0).hypot(qy.max(0.0));
    let inside = qx.max(qy).min(0.0);
    outside + inside <= radius
}

fn distance_to_segment(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let length_squared = dx * dx + dy * dy;
    let t = if length_squared == 0.0 {
        0.0
    } else {
        (((x - ax) * dx + (y - ay) * dy) / length_squared).clamp(0.0, 1.0)
    };
    (x - (ax + t * dx)).hypot(y - (ay + t * dy))
}

fn inside_mark(x: f64, y: f64) -> bool {
    const POINTS: [(f64, f64); 5] = [
        (10.0, 27.5),
        (10.0, 12.5),
        (20.0, 23.125),
        (30.0, 12.5),
        (30.0, 27.5),
    ];
    POINTS
        .windows(2)
        .any(|pair| distance_to_segment(x, y, pair[0].0, pair[0].1, pair[1].0, pair[1].1) <= 1.875)
}

fn sample(x: f64, y: f64) -> [u8; 4] {
    if !inside_background(x, y, 0.625) {
        return [0, 0, 0, 0];
    }
    if !inside_background(x, y, 0.9375) {
        return [222, 223, 218, 255];
    }
    if (x - 30.0).hypot(y - 27.5) <= 1.875 {
        return [184, 115, 67, 255];
    }
    if inside_mark(x, y) {
        return [32, 34, 35, 255];
    }
    [244, 244, 240, 255]
}

fn render(size: u32) -> Vec<u8> {
    const SAMPLES: u32 = 4;
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for py in 0..size {
        for px in 0..size {
            let mut alpha = 0_u32;
            let mut premultiplied = [0_u32; 3];
            for sy in 0..SAMPLES {
                for sx in 0..SAMPLES {
                    let x = (px as f64 + (sx as f64 + 0.5) / SAMPLES as f64) * 40.0 / size as f64;
                    let y = (py as f64 + (sy as f64 + 0.5) / SAMPLES as f64) * 40.0 / size as f64;
                    let value = sample(x, y);
                    alpha += value[3] as u32;
                    for channel in 0..3 {
                        premultiplied[channel] += value[channel] as u32 * value[3] as u32;
                    }
                }
            }
            let count = SAMPLES * SAMPLES;
            for value in premultiplied {
                pixels.push(if alpha == 0 { 0 } else { (value / alpha) as u8 });
            }
            pixels.push((alpha / count) as u8);
        }
    }
    pixels
}

fn write_png(path: &Path, size: u32) {
    let image = IcnsImage::from_data(PixelFormat::RGBA, size, size, render(size))
        .expect("invalid generated icon pixels");
    image
        .write_png(BufWriter::new(
            File::create(path).expect("failed to create icon PNG"),
        ))
        .expect("failed to write icon PNG");
}

fn write_icons(directory: &Path) {
    fs::create_dir_all(directory).expect("failed to create icon directory");
    write_png(&directory.join("32x32.png"), 32);
    write_png(&directory.join("128x128.png"), 128);
    write_png(&directory.join("256x256.png"), 256);

    let mut ico = IconDir::new(ResourceType::Icon);
    for size in [16, 24, 32, 48, 64, 128, 256] {
        let image = IconImage::from_rgba_data(size, size, render(size));
        ico.add_entry(IconDirEntry::encode_as_png(&image).expect("failed to encode ICO frame"));
    }
    ico.write(File::create(directory.join("icon.ico")).expect("failed to create ICO"))
        .expect("failed to write ICO");

    let mut family = IconFamily::new();
    for size in [16, 32, 64, 128, 256, 512, 1024] {
        let image = IcnsImage::from_data(PixelFormat::RGBA, size, size, render(size))
            .expect("invalid generated ICNS pixels");
        family
            .add_icon(&image)
            .expect("failed to encode ICNS frame");
    }
    family
        .write(BufWriter::new(
            File::create(directory.join("icon.icns")).expect("failed to create ICNS"),
        ))
        .expect("failed to write ICNS");
}

fn main() {
    let directory = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("icons");
    write_icons(&directory);
    tauri_build::build()
}
