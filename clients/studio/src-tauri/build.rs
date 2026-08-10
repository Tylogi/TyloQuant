use icns::{IconFamily, Image as IcnsImage, PixelFormat};
use ico::{IconDir, IconDirEntry, IconImage, ResourceType};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

const MARK: [(f64, f64); 12] = [
    (9.0, 27.0),
    (9.0, 12.0),
    (14.0, 12.0),
    (20.0, 20.25),
    (26.0, 12.0),
    (31.0, 12.0),
    (31.0, 27.0),
    (26.0, 27.0),
    (26.0, 19.8),
    (20.0, 27.5),
    (14.0, 19.8),
    (14.0, 27.0),
];

fn inside_mark(x: f64, y: f64) -> bool {
    let mut inside = false;
    let mut previous = MARK.len() - 1;
    for current in 0..MARK.len() {
        let (cx, cy) = MARK[current];
        let (px, py) = MARK[previous];
        if (cy > y) != (py > y) && x < (px - cx) * (y - cy) / (py - cy) + cx {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn inside_background(x: f64, y: f64) -> bool {
    let qx = (x - 20.0).abs() - 12.0;
    let qy = (y - 20.0).abs() - 12.0;
    let outside = qx.max(0.0).hypot(qy.max(0.0));
    let inside = qx.max(qy).min(0.0);
    outside + inside <= 8.0
}

fn inside_underline(x: f64, y: f64) -> bool {
    let nearest_x = x.clamp(11.0, 29.0);
    (x - nearest_x).hypot(y - 31.0) <= 1.5
}

fn sample(x: f64, y: f64) -> [u8; 4] {
    if !inside_background(x, y) {
        return [0, 0, 0, 0];
    }
    if inside_underline(x, y) {
        return [185, 133, 76, 255];
    }
    if inside_mark(x, y) {
        let t = (((x + y) * 0.5 - 8.0) / 23.5).clamp(0.0, 1.0);
        let channel = |start: f64, end: f64| (start + (end - start) * t).round() as u8;
        return [
            channel(154.0, 59.0),
            channel(106.0, 36.0),
            channel(58.0, 20.0),
            255,
        ];
    }
    [251, 247, 242, 255]
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
