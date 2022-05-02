extern crate image;

// 最近傍近似
pub fn resize_with_nearest_neighbor(img: &image::RgbImage, scale: f32) -> image::RgbImage {
    let w = img.width() as u32;
    let h = img.height() as u32; 
    let dst_w = (w as f32 * scale) as u32;
    let dst_h = (h as f32 * scale) as u32;
    let mut dst = image::RgbImage::new(dst_w, dst_h);
    for j in 0..dst_h {
        // 四捨五入する。
        let y = (j as f32 / scale + 0.5) as u32;
        let y = if y > h - 1 {h - 1} else {y};
        for i in 0..dst_w {
            // 四捨五入する。
            let x = (i as f32 / scale + 0.5) as u32;
            let x = if x > w - 1 {w - 1} else {x};
            let dp = img.get_pixel(x, y);
            dst.put_pixel(i, j, *dp);
        }
    }    
    return dst;
}

fn cap(v:u32, max:u32) -> u32 {
    if v >= max {max - 1} else {v}
} 

fn saturate<const N:usize>(p: &image::Rgb<f32>) -> u8 {
    let v = 255.0 * p[N];
    if v > 255.0 {255} else {v as u8}
}

fn convert_to_u8(p: &image::Rgb<f32>) -> image::Rgb<u8> {
    let r = saturate::<0>(p);
    let g = saturate::<1>(p);
    let b = saturate::<2>(p);
    image::Rgb([r, g, b])
}

fn add(lhs:&image::Rgb<f32>, rhs:&image::Rgb<f32>) -> image::Rgb<f32> {
    let r = lhs[0] + rhs[0];
    let g = lhs[1] + rhs[1];
    let b = lhs[2] + rhs[2];
    image::Rgb([r, g, b])
}

fn mul(a:f32, rhs:&image::Rgb<f32>) -> image::Rgb<f32> {
    let r = a * rhs[0];
    let g = a * rhs[1];
    let b = a * rhs[2];
    image::Rgb([r, g, b])
}

fn make_color_by_bilinear(y0:u32, y1:u32, x0:u32, x1:u32, s:f32, t:f32, img:&image::Rgb32FImage) -> image::Rgb<u8> {
    let a = img.get_pixel(x0, y0);
    let b = img.get_pixel(x1, y0);
    let c = img.get_pixel(x0, y1);
    let d = img.get_pixel(x1, y1);
     
    let p = add(&mul(t, &b), &mul(1.0 - t, &a));
    let q = add(&mul(t, &d), &mul(1.0 - t, &c)); 
    let r = add(&mul(1.0 - s, &p), &mul(s, &q));
    
    return convert_to_u8(&r);
}

// Bilinear補間
pub fn resize_with_bilinear(src: &image::Rgb32FImage, scale: f32) -> image::RgbImage {
    let (src_w, src_h) = src.dimensions();
    let dst_w = (src_w as f32 * scale) as u32;
    let dst_h = (src_h as f32 * scale) as u32;
    
    let mut dst = image::RgbImage::new(dst_w, dst_h);
    for j in 0..dst_h {
        let fy = j as f32 / scale; // 元画像における位置を計算(不動小数点)
        let mut y0 = fy as u32;
        y0 = cap(y0, src_h);
        let s = fy - (y0 as f32);
        let mut y1 = y0 + 1;
        y1 = cap(y1, src_h);
        for i in 0..dst_w {
            let fx = i as f32 / scale; // 元画像における位置を計算(不動小数点)
            let mut x0 = fx as u32;
            x0 = cap(x0, src_w);
            let t = fx - x0 as f32;
            let mut x1 = x0 + 1;
            x1 = cap(x1, src_w);
            let dp = make_color_by_bilinear(y0, y1, x0, x1, s, t, src);
            dst.put_pixel(i, j, dp);
        }
    }    
    return dst;
}